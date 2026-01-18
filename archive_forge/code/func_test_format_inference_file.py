import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_format_inference_file():
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
        img = Image.from_file(f)
        assert img.format == 'gif'