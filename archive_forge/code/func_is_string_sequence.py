import codecs
from collections import deque
import contextlib
import csv
from glob import iglob as std_iglob
import io
import json
import logging
import os
import py_compile
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from . import DistlibException
from .compat import (string_types, text_type, shutil, raw_input, StringIO,
def is_string_sequence(seq):
    result = True
    i = None
    for i, s in enumerate(seq):
        if not isinstance(s, string_types):
            result = False
            break
    assert i is not None
    return result