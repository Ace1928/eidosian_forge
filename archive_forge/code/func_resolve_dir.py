import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def resolve_dir(self, name):
    """
        If the name represents a directory, return that name
        as a directory (with the trailing slash).
        """
    names = self._name_set()
    dirname = name + '/'
    dir_match = name not in names and dirname in names
    return dirname if dir_match else name