from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def get_file_template(name, from_template):
    path = os.path.join(os.path.dirname(from_template.name), name)
    return from_template.__class__.from_filename(path, namespace=from_template.namespace, get_template=from_template.get_template)