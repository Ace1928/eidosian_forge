import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_non_callable_classmethod(self):

    class BadClassMethod:
        not_callable = classmethod(None)
    self.assertFalse(_callable(BadClassMethod.not_callable))