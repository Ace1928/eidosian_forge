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
def detect_start_response(status, headers, exc_info=None):
    try:
        return start_response(status, headers, exc_info)
    except:
        raise
    else:
        started.append(True)