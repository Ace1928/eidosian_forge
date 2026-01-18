from typing import List, Tuple, Dict
from numba import types
from numba.core import cgutils
from numba.core.extending import make_attribute_wrapper, models, register_model
from numba.core.imputils import Registry as ImplRegistry
from numba.core.typing.templates import ConcreteTemplate
from numba.core.typing.templates import Registry as TypingRegistry
from numba.core.typing.templates import signature
from numba.cuda import stubs
from numba.cuda.errors import CudaLoweringError
def make_lowering(fml_arg_list):
    """Meta function to create a lowering for the constructor. Flattens
        the arguments by converting vector_type into load instructions for each
        of its attributes. Such as float2 -> float2.x, float2.y.
        """

    def lowering(context, builder, sig, actual_args):
        source_list = []
        for argidx, fml_arg in enumerate(fml_arg_list):
            if isinstance(fml_arg, VectorType):
                pxy = cgutils.create_struct_proxy(fml_arg)(context, builder, actual_args[argidx])
                source_list += [getattr(pxy, attr) for attr in fml_arg.attr_names]
            else:
                source_list.append(actual_args[argidx])
        if len(source_list) != vector_type.num_elements:
            raise CudaLoweringError(f'Unmatched number of source elements ({len(source_list)}) and target elements ({{vector_type.num_elements}}).')
        out = cgutils.create_struct_proxy(vector_type)(context, builder)
        for attr_name, source in zip(vector_type.attr_names, source_list):
            setattr(out, attr_name, source)
        return out._getvalue()
    return lowering