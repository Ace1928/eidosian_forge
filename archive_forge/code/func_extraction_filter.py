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
def extraction_filter(member, path):
    """Run tarfile.tar_filter, but raise the expected ValueError"""
    try:
        return tarfile.tar_filter(member, path)
    except tarfile.FilterError as exc:
        raise ValueError(str(exc))