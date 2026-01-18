import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_non_callable_staticmethod(self):

    class BadStaticMethod:
        not_callable = staticmethod(None)
    self.assertFalse(_callable(BadStaticMethod.not_callable))