import contextlib
import fcntl
import itertools
import multiprocessing
import os
import pty
import re
import signal
import struct
import sys
import tempfile
import termios
import time
import traceback
import types
from typing import Optional, Generator, Tuple
import typing
@contextlib.contextmanager
def isolate(child_function: typing.Callable[[], int], stdin_pipe: bool=False, stdout_pipe: bool=False, stderr_pipe: bool=True, *, lines: int=LINES, columns: int=COLUMNS) -> Generator[IsolationEnvironment, None, None]:
    with _fifos('stdin' if stdin_pipe else None, 'stdout' if stdout_pipe else None, 'stderr' if stderr_pipe else None) as fifo_paths:
        result_r, result_w = os.pipe()
        env_pid, tty = pty.fork()
        if env_pid == pty.CHILD:
            try:
                os.close(result_r)
                pts = os.ttyname(pty.STDOUT_FILENO)
                ctx = multiprocessing.get_context('spawn')
                p = ctx.Process(target=_run_test, args=(child_function, pts, *fifo_paths))
                p.start()

                def handle_terminate(signum: int, frame: Optional[types.FrameType]) -> None:
                    if p.is_alive():
                        p.terminate()
                signal.signal(signal.SIGTERM, handle_terminate)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                p.join(2)
                if p.exitcode is None:
                    raise TimeoutException(p.pid)
            except BaseException:
                try:
                    with os.fdopen(result_w, 'w') as result_writer:
                        traceback.print_exc(file=result_writer)
                finally:
                    for path, write in zip(fifo_paths, [False, True, True]):
                        _open_fifo(path, write)
                    os._exit(1)
            with os.fdopen(result_w, 'w') as result_writer:
                result_writer.write(f'{p.exitcode}\n')
            os._exit(0)
        else:
            os.close(result_w)
            fcntl.ioctl(tty, termios.TIOCSWINSZ, struct.pack('HHHH', lines, columns, 0, 0))
            env = IsolationEnvironment(env_pid, tty, *fifo_paths)
            time.sleep(0.01)
            try:
                try:
                    yield env
                finally:

                    def get_return_code() -> int:
                        with os.fdopen(result_r) as result_reader:
                            result = result_reader.readline().rstrip()
                            try:
                                return int(result)
                            except ValueError:
                                pass
                            trace = result_reader.read()
                            raise TestProcessNotComplete('\n'.join([result, trace]))
                    env.close(get_return_code)
            except Exception:
                raw, processed = env.recorded_output()
                if raw:
                    print(f'Raw output: {repr(raw)}', file=sys.stderr)
                if processed:
                    print(f'Recorded output: {repr(processed)}', file=sys.stderr)
                raise