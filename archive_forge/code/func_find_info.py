import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
def find_info(f: NativeFunction) -> Tuple[Optional[Dict[str, DifferentiabilityInfo]], bool]:
    if 'generated' in f.tags and f.func.kind() == SchemaKind.out:
        return (None, False)
    if f.func in differentiability_infos:
        return (differentiability_infos[f.func], True)
    f_sig = f.func.signature(strip_default=True)
    if f_sig in functional_info_by_signature and (not is_foreach_func(f)):
        return (functional_info_by_signature[f_sig], False)
    if 'generated' in f.tags and f_sig in non_functional_info_by_signature:
        info_dict = non_functional_info_by_signature[f_sig]
        assert not any((any(('self' in str(inpt.nctype.name) for inpt in info.all_saved_inputs)) for info in info_dict.values())), f'''Attempted to convert a derivative formula for a mutable operator\n to be used by automatically by its functional variant ("{str(f.func)}").\n this is not currently supported (we'd need to fix up the formula in the codegen).'''
        return (info_dict, False)
    if is_foreach_func(f):
        assert f.func not in differentiability_infos
        diff_info, is_generated = gen_foreach_derivativeinfo(f, functional_info_by_signature, non_functional_info_by_signature)
        if diff_info is None:
            return (None, False)
        diff_info_dict = {'Default': diff_info}
        if is_generated:
            differentiability_infos[f.func] = diff_info_dict
            functional_info_by_signature[f.func] = diff_info_dict
        return (diff_info_dict, is_generated)
    return (None, False)