import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import
def getEmailAddress(self):
    email = os.environ.get('NEOS_EMAIL', '')
    if _email_re.match(email):
        return email
    raise RuntimeError("NEOS requires a valid email address. Please set the 'NEOS_EMAIL' environment variable.")