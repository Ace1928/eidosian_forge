from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
def process_ir_type(typ: Type, properties: 'LazyIrProperties', *, symint: bool) -> Union[BaseCType, VectorCType, OptionalCType, ListCType]:
    """
    This function takes a type from NativeFunctions and converts it for use with
    lazy tensor codegen.

    Type conversion for lazy currently consists of
     (1) changing at::Tensors into lazy::Values
     (2) wrapping everything in a BaseCType
     (3) making cpp-reference types into cpp-value types (e.g. vector instead of IntArrayRef)

    (1) converts at::Tensors to lazy::Values (which wrap lazy::Nodes, with which Lazy IR represents tensors.)
    There is special handling for Optional[Tensor] or List[Tensor], etc- hence 'tensor-like'

    This is incomplete- there are assertions in places that it's expected to need to add
    more types as the codegen is used with more operators.
    """
    if isinstance(typ, BaseType):
        if typ.name == BaseTy.Tensor:
            return BaseCType(getValueT())
        elif typ.name == BaseTy.Scalar:
            if properties.TreatScalarsAsConstants:
                return BaseCType(scalarT)
            return BaseCType(getValueT())
        elif typ.name == BaseTy.ScalarType:
            return BaseCType(scalarTypeT)
        elif typ.name == BaseTy.int:
            return BaseCType(longT)
        elif typ.name == BaseTy.SymInt:
            if symint:
                return BaseCType(getValueT())
            else:
                return BaseCType(longT)
        elif typ.name == BaseTy.bool:
            return BaseCType(boolT)
        elif typ.name == BaseTy.float:
            return BaseCType(doubleT)
        elif typ.name == BaseTy.str:
            return BaseCType(stringT)
        elif typ.name == BaseTy.Device:
            return BaseCType(deviceT)
        elif typ.name == BaseTy.Generator:
            return BaseCType(generatorT)
        elif typ.name == BaseTy.Layout:
            return BaseCType(layoutT)
        elif typ.name == BaseTy.MemoryFormat:
            return BaseCType(memoryFormatT)
        else:
            raise AssertionError(f'TODO add support for type {repr(typ)}')
    elif isinstance(typ, OptionalType):
        return OptionalCType(process_ir_type(typ.elem, properties, symint=symint))
    elif isinstance(typ, ListType):
        if str(typ.elem) == 'Tensor?':
            return ListCType(OptionalCType(BaseCType(getValueT())))
        elif str(typ.elem) == 'Tensor':
            return BaseCType(tensorListValueT)
        elif typ.elem == BaseType(BaseTy.SymInt):
            return VectorCType(BaseCType(longT))
        else:
            return VectorCType(process_ir_type(typ.elem, properties, symint=symint))
    else:
        raise AssertionError(f'unrecognized type {repr(typ)}')