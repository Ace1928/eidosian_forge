import pathlib
from panel.io.mime_render import (
def test_exec_with_return_multi_line():
    assert exec_with_return('a = 1\nb = 2\na + b') == 3