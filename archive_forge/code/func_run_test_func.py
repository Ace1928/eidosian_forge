import contextlib
import json
import sys
from StringIO import StringIO
import traceback
from google.appengine.api import app_identity
import google.auth
from google.auth import _helpers
from google.auth import app_engine
import google.auth.transport.urllib3
import urllib3.contrib.appengine
import webapp2
def run_test_func(func):
    with capture() as capsys:
        try:
            func()
            return (True, '')
        except Exception as exc:
            output = FAILED_TEST_TMPL.format(func.func_name, exc, traceback.format_exc(), capsys.getvalue())
            return (False, output)