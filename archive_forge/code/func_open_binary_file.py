from __future__ import (absolute_import, division, print_function)
import resource
import base64
import contextlib
import errno
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
def open_binary_file(path, mode='rb'):
    """Open the given path for binary access."""
    if 'b' not in mode:
        raise Exception('mode must include "b" for binary files: %s' % mode)
    return io.open(to_bytes(path), mode)