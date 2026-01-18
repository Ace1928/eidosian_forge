import pathlib
from panel.io.mime_render import (
def test_format_mime_None():
    assert format_mime(None) == ('None', 'text/plain')