import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def get_hash_hex(byte_str):
    m = hashlib.new('sha256')
    m.update(byte_str)
    return m.hexdigest()