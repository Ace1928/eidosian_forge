import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_jpeg_applies():
    assert JPG.applies(JPEG_FILE)
    assert JPG.applies(JPG_FILE)