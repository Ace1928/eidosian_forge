import asyncio
import asyncio.exceptions
import atexit
import errno
import os
import signal
import sys
import time
from subprocess import CalledProcessError
from threading import Thread
from traitlets import Any, Dict, List, default
from IPython.core import magic_arguments
from IPython.core.async_helpers import _AsyncIOProxy
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.utils.process import arg_split
@magic_arguments.magic_arguments()
@script_args
@cell_magic('script')
def shebang(self, line, cell):
    """Run a cell via a shell command

        The `%%script` line is like the #! line of script,
        specifying a program (bash, perl, ruby, etc.) with which to run.

        The rest of the cell is run by that program.

        Examples
        --------
        ::

            In [1]: %%script bash
               ...: for i in 1 2 3; do
               ...:   echo $i
               ...: done
            1
            2
            3
        """
    if self.event_loop is None:
        if sys.platform == 'win32':
            event_loop = asyncio.WindowsProactorEventLoopPolicy().new_event_loop()
        else:
            event_loop = asyncio.new_event_loop()
        self.event_loop = event_loop
        asyncio_thread = Thread(target=event_loop.run_forever, daemon=True)
        asyncio_thread.start()
    else:
        event_loop = self.event_loop

    def in_thread(coro):
        """Call a coroutine on the asyncio thread"""
        return asyncio.run_coroutine_threadsafe(coro, event_loop).result()

    async def _readchunk(stream):
        try:
            return await stream.readuntil(b'\n')
        except asyncio.exceptions.IncompleteReadError as e:
            return e.partial
        except asyncio.exceptions.LimitOverrunError as e:
            return await stream.read(e.consumed)

    async def _handle_stream(stream, stream_arg, file_object):
        while True:
            chunk = (await _readchunk(stream)).decode('utf8', errors='replace')
            if not chunk:
                break
            if stream_arg:
                self.shell.user_ns[stream_arg] = chunk
            else:
                file_object.write(chunk)
                file_object.flush()

    async def _stream_communicate(process, cell):
        process.stdin.write(cell)
        process.stdin.close()
        stdout_task = asyncio.create_task(_handle_stream(process.stdout, args.out, sys.stdout))
        stderr_task = asyncio.create_task(_handle_stream(process.stderr, args.err, sys.stderr))
        await asyncio.wait([stdout_task, stderr_task])
        await process.wait()
    argv = arg_split(line, posix=not sys.platform.startswith('win'))
    args, cmd = self.shebang.parser.parse_known_args(argv)
    try:
        p = in_thread(asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE))
    except OSError as e:
        if e.errno == errno.ENOENT:
            print("Couldn't find program: %r" % cmd[0])
            return
        else:
            raise
    if not cell.endswith('\n'):
        cell += '\n'
    cell = cell.encode('utf8', 'replace')
    if args.bg:
        self.bg_processes.append(p)
        self._gc_bg_processes()
        to_close = []
        if args.out:
            self.shell.user_ns[args.out] = _AsyncIOProxy(p.stdout, event_loop)
        else:
            to_close.append(p.stdout)
        if args.err:
            self.shell.user_ns[args.err] = _AsyncIOProxy(p.stderr, event_loop)
        else:
            to_close.append(p.stderr)
        event_loop.call_soon_threadsafe(lambda: asyncio.Task(self._run_script(p, cell, to_close)))
        if args.proc:
            proc_proxy = _AsyncIOProxy(p, event_loop)
            proc_proxy.stdout = _AsyncIOProxy(p.stdout, event_loop)
            proc_proxy.stderr = _AsyncIOProxy(p.stderr, event_loop)
            self.shell.user_ns[args.proc] = proc_proxy
        return
    try:
        in_thread(_stream_communicate(p, cell))
    except KeyboardInterrupt:
        try:
            p.send_signal(signal.SIGINT)
            in_thread(asyncio.wait_for(p.wait(), timeout=0.1))
            if p.returncode is not None:
                print('Process is interrupted.')
                return
            p.terminate()
            in_thread(asyncio.wait_for(p.wait(), timeout=0.1))
            if p.returncode is not None:
                print('Process is terminated.')
                return
            p.kill()
            print('Process is killed.')
        except OSError:
            pass
        except Exception as e:
            print('Error while terminating subprocess (pid=%i): %s' % (p.pid, e))
        return
    if args.raise_error and p.returncode != 0:
        rc = p.returncode or -9
        raise CalledProcessError(rc, cell)