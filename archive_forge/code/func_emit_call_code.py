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
def emit_call_code(self):
    params = self.in_params + self.out_params
    return '{op_name}({params});'.format(op_name=self.name, params=', '.join([var.lvar_name for var in params]))