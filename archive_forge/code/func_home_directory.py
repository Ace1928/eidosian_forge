import logging
import os
import secrets
import shutil
import tempfile
import uuid
from contextlib import suppress
from urllib.parse import quote
import requests
from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, tokenize
def home_directory(self):
    """Get user's home directory"""
    out = self._call('GETHOMEDIRECTORY')
    return out.json()['Path']