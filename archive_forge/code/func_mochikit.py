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
def mochikit(self, environ, start_response):
    """
        Static path where MochiKit lives
        """
    app = urlparser.StaticURLParser(os.path.join(os.path.dirname(__file__), 'mochikit'))
    return app(environ, start_response)