import re
import sys
import types
import unittest
from tensorflow.python.eager import def_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
@def_function.function(autograph=False)
def fn_wrapper():
    self.assertions = []
    self.raises_cm = None
    self.graph_assertions = []
    self.trace_log = []
    fn()
    targets = [args for _, args in self.assertions]
    return targets