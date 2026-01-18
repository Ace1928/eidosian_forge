from typing import Dict, Optional, Set
import torch
from torch._ops import OpOverload, OpOverloadPacket, HigherOrderOperator
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBase
def get_view_copy_of_view_op(schema: torch._C.FunctionSchema) -> Optional[OpOverload]:
    if is_view_op(schema) and schema.name.startswith('aten::'):
        view_op_name = schema.name.split('::')[1]
        view_op_overload = schema.overload_name if schema.overload_name != '' else 'default'
        view_copy_op_name = view_op_name + '_copy'
        if not hasattr(torch.ops.aten, view_copy_op_name):
            raise InternalError(f'{schema.name} is missing a view_copy variant')
        view_copy_op_overload_packet = getattr(torch.ops.aten, view_copy_op_name)
        if not hasattr(view_copy_op_overload_packet, view_op_overload):
            raise InternalError(f'{schema.name} is missing a view_copy variant')
        return getattr(view_copy_op_overload_packet, view_op_overload)
    return None