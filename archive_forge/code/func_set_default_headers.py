import re
import inspect
import traceback
import copy
import logging
import hmac
from base64 import b64decode
import tornado
from ..utils import template, bugreport, strtobool
def set_default_headers(self):
    if not (self.application.options.basic_auth or self.application.options.auth):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with,access-control-allow-origin,authorization,content-type')
        self.set_header('Access-Control-Allow-Methods', ' PUT, DELETE, OPTIONS, POST, GET, PATCH')