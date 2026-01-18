import html
import sys
import os
import traceback
from io import StringIO
import pprint
import itertools
import time
import re
from paste.exceptions import errormiddleware, formatter, collector
from paste import wsgilib
from paste import urlparser
from paste import httpexceptions
from paste import registry
from paste import request
from paste import response
from paste.evalexception import evalcontext
def eval_javascript(self):
    base_path = self.base_path + '/_debug'
    return '<script type="text/javascript" src="%s/media/MochiKit.packed.js"></script>\n<script type="text/javascript" src="%s/media/debug.js"></script>\n<script type="text/javascript">\ndebug_base = %r;\ndebug_count = %r;\n</script>\n' % (base_path, base_path, base_path, self.counter)