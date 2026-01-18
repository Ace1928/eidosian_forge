from llvmlite import ir
from llvmlite import binding as ll
from numba.core import datamodel
import unittest
def test_as_return(self):
    """
        - Is as_return() and from_return() implemented?
        - Are they the inverse of each other?
        """
    fnty = ir.FunctionType(ir.VoidType(), [])
    function = ir.Function(self.module, fnty, name='test_as_return')
    builder = ir.IRBuilder()
    builder.position_at_end(function.append_basic_block())
    undef_value = ir.Constant(self.datamodel.get_value_type(), None)
    ret = self.datamodel.as_return(builder, undef_value)
    self.assertIsNot(ret, NotImplemented, 'as_return returned NotImplementedError')
    self.assertEqual(ret.type, self.datamodel.get_return_type())
    rev_value = self.datamodel.from_return(builder, ret)
    self.assertEqual(rev_value.type, self.datamodel.get_value_type())
    builder.ret_void()
    materialized = ll.parse_assembly(str(self.module))
    str(materialized)