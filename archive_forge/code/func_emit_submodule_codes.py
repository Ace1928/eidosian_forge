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
def emit_submodule_codes(self):
    """Returns a CUDA device function code.

        The emitted code assumes that ``block_stride`` and `blockDim.x` is a
        power of 2.
        """
    in_param, = self.in_params
    out_param, = self.out_params
    op_name = '{}_op'.format(self.name)
    postmap_name = '{}_postmap'.format(self.name)
    template = string.Template('\n#define ${op_name}(a, b) (${reduce_expr})\n#define ${postmap_name}(a, out0) (${postmap_cast})\n\ntemplate <typename InType, typename OutType, typename InIndexerType, typename OutIndexerType>\n__device__ void ${name}(\n        InType in_arr, OutType out_arr,\n        InIndexerType in_ind, OutIndexerType out_ind, int block_stride) {\n    typedef ${in_type} type_in0_raw;\n    typedef ${out_type} type_out0_raw;\n    typedef ${reduce_ctype} _type_reduce;\n    extern __shared__ char _sdata_raw[];\n    _type_reduce *sdata = reinterpret_cast<_type_reduce*>(_sdata_raw);\n    unsigned int tid = threadIdx.x;\n    int _J = tid >> __popc(block_stride - 1);\n    ptrdiff_t _j = (ptrdiff_t)_J * out_ind.size();\n    int J_stride = blockDim.x >> __popc(block_stride - 1);\n    ptrdiff_t j_stride = (ptrdiff_t)J_stride * out_ind.size();\n\n    for (ptrdiff_t _i = (ptrdiff_t)blockIdx.x * block_stride; _i < out_ind.size(); _i += (ptrdiff_t)gridDim.x * block_stride) {\n        _type_reduce s = _type_reduce(${identity});\n        ptrdiff_t i = _i + (tid & (block_stride - 1));\n        for (ptrdiff_t j = i + _j; j < in_ind.size(); j += j_stride) {\n            in_ind.set(j);\n            s = ${op_name}(s, static_cast<_type_reduce>(in_arr[in_ind.get()]));\n        }\n        sdata[tid] = s;\n        __syncthreads();\n        for (unsigned int block = blockDim.x / 2; block >= block_stride; block >>= 1) {\n            if (tid < block) {\n                sdata[tid] = ${op_name}(sdata[tid], sdata[tid + block]);\n            }\n            __syncthreads();\n        }\n        if (tid < block_stride) {\n            s = sdata[tid];\n        }\n        if (tid < block_stride && i < out_ind.size()) {\n            out_ind.set(i);\n            ${postmap_name}(s, out_arr[out_ind.get()]);\n        }\n        __syncthreads();\n    }\n}')
    code = template.substitute(name=self.name, op_name=op_name, postmap_name=postmap_name, in_type=get_typename(in_param.dtype), out_type=get_typename(out_param.dtype), reduce_ctype=self.reduce_ctype, reduce_expr=self.expr, identity=self.identity, postmap_cast=self.postmap_cast_code)
    return [code]