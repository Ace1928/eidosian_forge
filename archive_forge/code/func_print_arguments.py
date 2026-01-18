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
def print_arguments():
    print()
    print('<H3>Command Line Arguments:</H3>')
    print()
    print(sys.argv)
    print()