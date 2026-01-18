import pathlib
from panel.io.mime_render import (
def test_exec_captures_print():

    def capture_stdout(stdout):
        assert stdout == 'foo'
    stdout = WriteCallbackStream(capture_stdout)
    assert exec_with_return('print("foo")', stdout=stdout) is None
    assert stdout.getvalue() == 'foo'