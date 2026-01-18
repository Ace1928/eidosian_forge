import os
import sys
import time
import random
import os.path
import platform
import warnings
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import libcloud.security
from libcloud.http import LibcloudConnection
from libcloud.test import unittest, no_network
from libcloud.utils.py3 import reload, httplib, assertRaisesRegex
def test_custom_ca_path_using_env_var_doesnt_exist(self):
    os.environ['SSL_CERT_FILE'] = '/foo/doesnt/exist'
    try:
        reload(libcloud.security)
    except ValueError as e:
        msg = "Certificate file /foo/doesnt/exist doesn't exist"
        self.assertEqual(str(e), msg)
    else:
        self.fail('Exception was not thrown')