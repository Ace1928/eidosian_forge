import unittest
import cupy.testing._parameterized
def param_name(_, i, param):
    return str(i)