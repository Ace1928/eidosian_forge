import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_image_format():
    Image(format='png')
    Image(format='jpeg')
    Image(format='url')