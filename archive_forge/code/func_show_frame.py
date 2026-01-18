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
def show_frame(self, tbid, debug_info, **kw):
    frame = debug_info.frame(int(tbid))
    vars = frame.tb_frame.f_locals
    if vars:
        registry.restorer.restoration_begin(debug_info.counter)
        local_vars = make_table(vars)
        registry.restorer.restoration_end()
    else:
        local_vars = 'No local vars'
    return input_form(tbid, debug_info) + local_vars