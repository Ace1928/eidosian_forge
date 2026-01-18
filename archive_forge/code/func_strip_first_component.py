from __future__ import annotations
import contextlib
import functools
import operator
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from typing import Iterator
def strip_first_component(member: tarfile.TarInfo, path) -> tarfile.TarInfo:
    _, member.name = member.name.split('/', 1)
    return member