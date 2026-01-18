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
def in_venv():
    if hasattr(sys, 'real_prefix'):
        result = True
    else:
        result = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
    return result