from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
def system_multicall(self, call_list):
    """system.multicall([{'methodName': 'add', 'params': [2, 2]}, ...]) => [[4], ...]

        Allows the caller to package multiple XML-RPC calls into a single
        request.

        See http://www.xmlrpc.com/discuss/msgReader$1208
        """
    results = []
    for call in call_list:
        method_name = call['methodName']
        params = call['params']
        try:
            results.append([self._dispatch(method_name, params)])
        except Fault as fault:
            results.append({'faultCode': fault.faultCode, 'faultString': fault.faultString})
        except BaseException as exc:
            results.append({'faultCode': 1, 'faultString': '%s:%s' % (type(exc), exc)})
    return results