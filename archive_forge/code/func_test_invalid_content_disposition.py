from tornado.httputil import (
from tornado.escape import utf8, native_str
from tornado.log import gen_log
from tornado.testing import ExpectLog
from tornado.test.util import ignore_deprecation
import copy
import datetime
import logging
import pickle
import time
import urllib.parse
import unittest
from typing import Tuple, Dict, List
def test_invalid_content_disposition(self):
    data = b'--1234\nContent-Disposition: invalid; name="files"; filename="ab.txt"\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
    args, files = form_data_args()
    with ExpectLog(gen_log, 'Invalid multipart/form-data'):
        parse_multipart_form_data(b'1234', data, args, files)
    self.assertEqual(files, {})