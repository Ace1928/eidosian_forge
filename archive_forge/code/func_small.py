import inspect
import keyword
import linecache
import os
import pydoc
import sys
import tempfile
import time
import tokenize
import traceback
import warnings
from html import escape as html_escape
def small(text):
    if text:
        return '<small>' + text + '</small>'
    else:
        return ''