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
def showtopic(self, topic, more_xrefs=''):
    try:
        import pydoc_data.topics
    except ImportError:
        self.output.write('\nSorry, topic and keyword documentation is not available because the\nmodule "pydoc_data.topics" could not be found.\n')
        return
    target = self.topics.get(topic, self.keywords.get(topic))
    if not target:
        self.output.write('no documentation found for %s\n' % repr(topic))
        return
    if type(target) is type(''):
        return self.showtopic(target, more_xrefs)
    label, xrefs = target
    try:
        doc = pydoc_data.topics.topics[label]
    except KeyError:
        self.output.write('no documentation found for %s\n' % repr(topic))
        return
    doc = doc.strip() + '\n'
    if more_xrefs:
        xrefs = (xrefs or '') + ' ' + more_xrefs
    if xrefs:
        import textwrap
        text = 'Related help topics: ' + ', '.join(xrefs.split()) + '\n'
        wrapped_text = textwrap.wrap(text, 72)
        doc += '\n%s\n' % '\n'.join(wrapped_text)
    pager(doc)