import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_image_caption(document, comm):
    png = PNG(PNG_FILE, caption='Some Caption')
    model = png.get_root(document, comm)
    assert 'Some Caption' in model.text
    assert 'figcaption' in model.text