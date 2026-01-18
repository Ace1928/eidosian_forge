import pathlib
from panel.io.mime_render import (
def test_exec_with_return_None():
    assert exec_with_return('None') is None