import pathlib
from panel.io.mime_render import (
def test_exec_with_return_no_return():
    assert exec_with_return('a = 1') is None