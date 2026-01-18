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
def remove_folder(self, folder):
    """Delete the named folder, which must be empty."""
    path = os.path.join(self._path, folder)
    entries = os.listdir(path)
    if entries == ['.mh_sequences']:
        os.remove(os.path.join(path, '.mh_sequences'))
    elif entries == []:
        pass
    else:
        raise NotEmptyError('Folder not empty: %s' % self._path)
    os.rmdir(path)