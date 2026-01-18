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
def print_form(form):
    """Dump the contents of a form as HTML."""
    keys = sorted(form.keys())
    print()
    print('<H3>Form Contents:</H3>')
    if not keys:
        print('<P>No form fields.')
    print('<DL>')
    for key in keys:
        print('<DT>' + html.escape(key) + ':', end=' ')
        value = form[key]
        print('<i>' + html.escape(repr(type(value))) + '</i>')
        print('<DD>' + html.escape(repr(value)))
    print('</DL>')
    print()