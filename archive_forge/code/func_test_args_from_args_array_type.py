import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
def test_args_from_args_array_type(self):
    fake_type = ArrayType(MyBaseType)
    fd = FunctionDefinition(FunctionDefinition)
    fd.arguments.append(FunctionArgument('fake-arg', fake_type, True, []))
    try:
        args_from_args(fd, [['invalid-argument']], {})
    except InvalidInput as e:
        assert ArrayType.__name__ in str(e)
    else:
        self.fail('Should have thrown an InvalidInput')