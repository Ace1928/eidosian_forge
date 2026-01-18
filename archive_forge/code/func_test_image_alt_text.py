import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_image_alt_text(document, comm):
    """Tests the loading of a image from a url"""
    url = 'https://raw.githubusercontent.com/holoviz/panel/main/doc/_static/logo.png'
    image_pane = PNG(url, embed=False, alt_text='Some alt text')
    model = image_pane.get_root(document, comm)
    assert 'alt=&#x27;Some alt text&#x27;' in model.text