from typing import List, Optional, Sequence, Set, Union
from torchgen import local
from torchgen.api.types import (
from torchgen.model import (
from torchgen.utils import assert_never
from .types import (
def returntype_type(t: Type, *, mutable: bool) -> CType:
    r = valuetype_type(t, binds='__placeholder__')
    if r is not None:
        return r.type
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                if local.use_const_ref_for_mutable_tensors():
                    return ConstRefCType(BaseCType(tensorT))
                else:
                    return MutRefCType(BaseCType(tensorT))
            else:
                return BaseCType(tensorT)
        elif t.name == BaseTy.Scalar:
            return BaseCType(scalarT)
    elif isinstance(t, ListType):
        assert not mutable, 'Native functions should never return a mutable tensor list. They should return void.'
        elem = returntype_type(t.elem, mutable=False)
        assert t.size is None, f'fixed size list returns not supported: {t}'
        return VectorCType(elem)
    raise AssertionError(f'unrecognized return type {t}')