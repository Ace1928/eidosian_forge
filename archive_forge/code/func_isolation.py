import contextlib
import io
import os
import shlex
import shutil
import sys
import tempfile
import typing as t
from types import TracebackType
from . import formatting
from . import termui
from . import utils
from ._compat import _find_binary_reader
@contextlib.contextmanager
def isolation(self, input: t.Optional[t.Union[str, bytes, t.IO[t.Any]]]=None, env: t.Optional[t.Mapping[str, t.Optional[str]]]=None, color: bool=False) -> t.Iterator[t.Tuple[io.BytesIO, t.Optional[io.BytesIO]]]:
    """A context manager that sets up the isolation for invoking of a
        command line tool.  This sets up stdin with the given input data
        and `os.environ` with the overrides from the given dictionary.
        This also rebinds some internals in Click to be mocked (like the
        prompt functionality).

        This is automatically done in the :meth:`invoke` method.

        :param input: the input stream to put into sys.stdin.
        :param env: the environment overrides as dictionary.
        :param color: whether the output should contain color codes. The
                      application can still override this explicitly.

        .. versionchanged:: 8.0
            ``stderr`` is opened with ``errors="backslashreplace"``
            instead of the default ``"strict"``.

        .. versionchanged:: 4.0
            Added the ``color`` parameter.
        """
    bytes_input = make_input_stream(input, self.charset)
    echo_input = None
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_forced_width = formatting.FORCED_WIDTH
    formatting.FORCED_WIDTH = 80
    env = self.make_env(env)
    bytes_output = io.BytesIO()
    if self.echo_stdin:
        bytes_input = echo_input = t.cast(t.BinaryIO, EchoingStdin(bytes_input, bytes_output))
    sys.stdin = text_input = _NamedTextIOWrapper(bytes_input, encoding=self.charset, name='<stdin>', mode='r')
    if self.echo_stdin:
        text_input._CHUNK_SIZE = 1
    sys.stdout = _NamedTextIOWrapper(bytes_output, encoding=self.charset, name='<stdout>', mode='w')
    bytes_error = None
    if self.mix_stderr:
        sys.stderr = sys.stdout
    else:
        bytes_error = io.BytesIO()
        sys.stderr = _NamedTextIOWrapper(bytes_error, encoding=self.charset, name='<stderr>', mode='w', errors='backslashreplace')

    @_pause_echo(echo_input)
    def visible_input(prompt: t.Optional[str]=None) -> str:
        sys.stdout.write(prompt or '')
        val = text_input.readline().rstrip('\r\n')
        sys.stdout.write(f'{val}\n')
        sys.stdout.flush()
        return val

    @_pause_echo(echo_input)
    def hidden_input(prompt: t.Optional[str]=None) -> str:
        sys.stdout.write(f'{prompt or ''}\n')
        sys.stdout.flush()
        return text_input.readline().rstrip('\r\n')

    @_pause_echo(echo_input)
    def _getchar(echo: bool) -> str:
        char = sys.stdin.read(1)
        if echo:
            sys.stdout.write(char)
        sys.stdout.flush()
        return char
    default_color = color

    def should_strip_ansi(stream: t.Optional[t.IO[t.Any]]=None, color: t.Optional[bool]=None) -> bool:
        if color is None:
            return not default_color
        return not color
    old_visible_prompt_func = termui.visible_prompt_func
    old_hidden_prompt_func = termui.hidden_prompt_func
    old__getchar_func = termui._getchar
    old_should_strip_ansi = utils.should_strip_ansi
    termui.visible_prompt_func = visible_input
    termui.hidden_prompt_func = hidden_input
    termui._getchar = _getchar
    utils.should_strip_ansi = should_strip_ansi
    old_env = {}
    try:
        for key, value in env.items():
            old_env[key] = os.environ.get(key)
            if value is None:
                try:
                    del os.environ[key]
                except Exception:
                    pass
            else:
                os.environ[key] = value
        yield (bytes_output, bytes_error)
    finally:
        for key, value in old_env.items():
            if value is None:
                try:
                    del os.environ[key]
                except Exception:
                    pass
            else:
                os.environ[key] = value
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin
        termui.visible_prompt_func = old_visible_prompt_func
        termui.hidden_prompt_func = old_hidden_prompt_func
        termui._getchar = old__getchar_func
        utils.should_strip_ansi = old_should_strip_ansi
        formatting.FORCED_WIDTH = old_forced_width