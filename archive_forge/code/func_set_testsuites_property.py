import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
@classmethod
def set_testsuites_property(cls, key, value):
    cls._testsuites_properties[key] = value