import pathlib
from panel.io.mime_render import (
def test_format_mime_repr_png():
    img, mime_type = format_mime(PNG())
    assert mime_type == 'text/html'
    assert img.startswith('<img src="data:image/png')