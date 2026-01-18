import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
def test_int_nonzero(self):
    self.assertEqual(1, from_param(int, 1))
    self.assertEqual(1, from_param(int, '1'))