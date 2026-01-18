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
def get_package_data(name, version):
    url = '%s/%s/package-%s.json' % (name[0].upper(), name, version)
    url = urljoin(_external_data_base_url, url)
    return _get_external_data(url)