import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
def test_from_params_dict_unset(self):
    assert from_params(DictType(int, str), {}, 'a', set()) is Unset