from typing import List, Optional, Sequence, Set, Union
from torchgen import local
from torchgen.api.types import (
from torchgen.model import (
from torchgen.utils import assert_never
from .types import (
def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName, remove_non_owning_ref_types: bool=False) -> NamedCType:
    r = valuetype_type(t, binds=binds, remove_non_owning_ref_types=remove_non_owning_ref_types)
    if r is not None:
        return r
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable and (not local.use_const_ref_for_mutable_tensors()):
                return NamedCType(binds, MutRefCType(BaseCType(tensorT)))
            else:
                return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f'base type should have been value type {t}')
    elif isinstance(t, OptionalType):
        if str(t.elem) == 'Tensor':
            if mutable and (not local.use_const_ref_for_mutable_tensors()):
                return NamedCType(binds, MutRefCType(BaseCType(tensorT)))
            else:
                return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(tensorT))))
        elif str(t.elem) == 'Scalar':
            return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if str(t.elem) == 'Tensor':
            return NamedCType(binds, BaseCType(tensorListT))
        elif str(t.elem) == 'Dimname':
            raise NotImplementedError("Executorch doesn't support Dimname")
        elif str(t.elem) == 'Tensor?':
            return NamedCType(binds, ArrayRefCType(OptionalCType(BaseCType(tensorT))))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        raise AssertionError(f'unrecognized type {repr(t)}')