import string
import numpy
from cupy._core import _codeblock
from cupy._core._fusion_variable import _TraceVariable
from cupy._core._fusion_variable import _TraceArray
from cupy._core._fusion_variable import _VariableSet
from cupy._core import _fusion_thread_local
from cupy._core import _kernel
from cupy._core import _reduction
from cupy._core._scalar import get_typename
def emit_code(self):
    _fusion_thread_local.check_not_runtime()
    assert len(self.in_params) == 1
    assert len(self.out_params) == 1
    in_param = list(self.in_params)[0]
    out_param = list(self.out_params)[0]
    params = ', '.join([in_param.var_name, out_param.var_name, in_param.indexer_name, out_param.indexer_name])
    return '{}({}, {});'.format(self.name, params, self.block_stride_name)