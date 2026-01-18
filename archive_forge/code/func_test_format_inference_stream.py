import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_format_inference_stream():
    fstream = io.BytesIO(b'')
    img = Image.from_file(fstream)
    assert img.format == 'png'