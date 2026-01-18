from io import StringIO, BytesIO, TextIOWrapper
from collections.abc import Mapping
import sys
import os
import urllib.parse
from email.parser import FeedParser
from email.message import Message
import html
import locale
import tempfile
import warnings
def print_directory():
    """Dump the current directory as HTML."""
    print()
    print('<H3>Current Working Directory:</H3>')
    try:
        pwd = os.getcwd()
    except OSError as msg:
        print('OSError:', html.escape(str(msg)))
    else:
        print(html.escape(pwd))
    print()