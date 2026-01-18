import pathlib
from panel.io.mime_render import (
def test_format_mime_str_with_escapes():
    assert format_mime('foo>bar') == ('foo&gt;bar', 'text/plain')