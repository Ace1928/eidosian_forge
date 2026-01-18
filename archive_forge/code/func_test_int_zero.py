import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
def test_int_zero(self):
    self.assertEqual(0, from_param(int, 0))
    self.assertEqual(0, from_param(int, '0'))