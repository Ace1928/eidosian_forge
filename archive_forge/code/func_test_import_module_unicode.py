from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
def test_import_module_unicode(self):
    self.assertIs(import_object('tornado.escape'), tornado.escape)