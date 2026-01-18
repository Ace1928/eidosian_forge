import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
def test_from_params_native_array(self):

    class params(dict):

        def getall(self, path):
            return ['1', '2']
    p = params({'a': []})
    assert from_params(ArrayType(int), p, 'a', set()) == [1, 2]