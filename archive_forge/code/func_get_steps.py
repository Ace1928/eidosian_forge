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
def get_steps(self, final):
    if not self.is_step(final):
        raise ValueError('Unknown: %r' % final)
    result = []
    todo = []
    seen = set()
    todo.append(final)
    while todo:
        step = todo.pop(0)
        if step in seen:
            if step != final:
                result.remove(step)
                result.append(step)
        else:
            seen.add(step)
            result.append(step)
            preds = self._preds.get(step, ())
            todo.extend(preds)
    return reversed(result)