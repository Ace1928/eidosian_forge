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
def html_navbar():
    version = html.escape('%s [%s, %s]' % (platform.python_version(), platform.python_build()[0], platform.python_compiler()))
    return '\n            <div style=\'float:left\'>\n                Python %s<br>%s\n            </div>\n            <div style=\'float:right\'>\n                <div style=\'text-align:center\'>\n                  <a href="index.html">Module Index</a>\n                  : <a href="topics.html">Topics</a>\n                  : <a href="keywords.html">Keywords</a>\n                </div>\n                <div>\n                    <form action="get" style=\'display:inline;\'>\n                      <input type=text name=key size=15>\n                      <input type=submit value="Get">\n                    </form>&nbsp;\n                    <form action="search" style=\'display:inline;\'>\n                      <input type=text name=key size=15>\n                      <input type=submit value="Search">\n                    </form>\n                </div>\n            </div>\n            ' % (version, html.escape(platform.platform(terse=True)))