import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def list_folders(self):
    """Return a list of folder names."""
    result = []
    for entry in os.listdir(self._path):
        if os.path.isdir(os.path.join(self._path, entry)):
            result.append(entry)
    return result