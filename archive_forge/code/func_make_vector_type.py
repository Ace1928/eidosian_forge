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
def make_vector_type(name: str, base_type: types.Type, attr_names: Tuple[str, ...], user_facing_object) -> types.Type:
    """Create a vector type.

    Parameters
    ----------
    name: str
        The name of the type.
    base_type: numba.types.Type
        The primitive type for each element in the vector.
    attr_names: tuple of str
        Name for each attribute.
    user_facing_object: object
        The handle to be used in cuda kernel.
    """

    class _VectorType(VectorType):
        """Internal instantiation of VectorType."""
        pass

    class VectorTypeModel(models.StructModel):

        def __init__(self, dmm, fe_type):
            members = [(attr_name, base_type) for attr_name in attr_names]
            super().__init__(dmm, fe_type, members)
    vector_type = _VectorType(name, base_type, attr_names, user_facing_object)
    register_model(_VectorType)(VectorTypeModel)
    for attr_name in attr_names:
        make_attribute_wrapper(_VectorType, attr_name, attr_name)
    return vector_type