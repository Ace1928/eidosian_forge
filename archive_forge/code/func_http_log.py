import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
def http_log(self, args, kwargs, resp, body):
    if not RDC_PP:
        self.simple_log(args, kwargs, resp, body)
    else:
        self.pretty_log(args, kwargs, resp, body)