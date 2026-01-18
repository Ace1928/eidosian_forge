import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def get_html_page(url):
    """Generate an HTML page for url."""
    complete_url = url
    if url.endswith('.html'):
        url = url[:-5]
    try:
        if url in ('', 'index'):
            title, content = html_index()
        elif url == 'topics':
            title, content = html_topics()
        elif url == 'keywords':
            title, content = html_keywords()
        elif '=' in url:
            op, _, url = url.partition('=')
            if op == 'search?key':
                title, content = html_search(url)
            elif op == 'topic?key':
                try:
                    title, content = html_topicpage(url)
                except ValueError:
                    title, content = html_getobj(url)
            elif op == 'get?key':
                if url in ('', 'index'):
                    title, content = html_index()
                else:
                    try:
                        title, content = html_getobj(url)
                    except ValueError:
                        title, content = html_topicpage(url)
            else:
                raise ValueError('bad pydoc url')
        else:
            title, content = html_getobj(url)
    except Exception as exc:
        title, content = html_error(complete_url, exc)
    return html.page(title, content)