import pathlib
from panel.io.mime_render import (
def test_format_mime_repr_html():
    assert format_mime(HTML('<b>BOLD</b>')) == ('<b>BOLD</b>', 'text/html')