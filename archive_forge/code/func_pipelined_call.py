import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
@wraps(__call__)
def pipelined_call(*args, **kwargs):
    from ..operation.element import factory, method as method_op
    from .data import Dataset, MultiDimensionalMapping
    inst = args[0]
    if not hasattr(inst._obj, '_pipeline'):
        return __call__(*args, **kwargs)
    inst_pipeline = copy.copy(inst._obj._pipeline)
    in_method = inst._obj._in_method
    if not in_method:
        inst._obj._in_method = True
    try:
        result = __call__(*args, **kwargs)
        if not in_method:
            init_op = factory.instance(output_type=type(inst), kwargs={'mode': getattr(inst, 'mode', None)})
            call_op = method_op.instance(input_type=type(inst), method_name='__call__', args=list(args[1:]), kwargs=kwargs)
            if isinstance(result, Dataset):
                result._pipeline = inst_pipeline.instance(operations=inst_pipeline.operations + [init_op, call_op], output_type=type(result))
            elif isinstance(result, MultiDimensionalMapping):
                for key, element in result.items():
                    getitem_op = method_op.instance(input_type=type(result), method_name='__getitem__', args=[key])
                    element._pipeline = inst_pipeline.instance(operations=inst_pipeline.operations + [init_op, call_op, getitem_op], output_type=type(result))
    finally:
        if not in_method:
            inst._obj._in_method = False
    return result