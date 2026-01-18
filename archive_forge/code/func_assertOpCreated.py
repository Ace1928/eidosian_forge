import re
import sys
import types
import unittest
from tensorflow.python.eager import def_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
def assertOpCreated(self, op_type):
    self.assertIn(op_type, self.trace_log)