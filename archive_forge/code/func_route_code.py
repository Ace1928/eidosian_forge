import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def route_code(self, route_code):
    """Adjust route_code on the way through."""
    if route_code is None:
        return self.routing_code
    return self.routing_code + '/' + route_code