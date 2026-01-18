import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_image_value():
    random_bytes = b'\x0ee\xca\x80\xcd\x9ak#\x7f\x07\x03\xa7'
    Image(value=random_bytes)