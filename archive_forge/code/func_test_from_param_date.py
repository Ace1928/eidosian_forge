import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
def test_from_param_date(self):
    assert from_param(datetime.date, '2008-02-28') == datetime.date(2008, 2, 28)