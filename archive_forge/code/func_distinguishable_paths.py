from collections import defaultdict, deque
from typing import Any, Counter, Dict, Iterable, Iterator, List, \
from ..exceptions import XMLSchemaValueError
from ..aliases import ModelGroupType, ModelParticleType, SchemaElementType
from ..translation import gettext as _
from .. import limits
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError
from .wildcards import XsdAnyElement, Xsd11AnyElement
from . import groups
def distinguishable_paths(path1: List[ModelParticleType], path2: List[ModelParticleType]) -> bool:
    """
    Checks if two model paths are distinguishable in a deterministic way, without looking forward
    or backtracking. The arguments are lists containing paths from the base group of the model to
    a couple of leaf elements. Returns `True` if there is a deterministic separation between paths,
    `False` if the paths are ambiguous.
    """
    e: ModelParticleType
    for k, e in enumerate(path1):
        if e not in path2:
            if not k:
                return True
            depth = k - 1
            break
    else:
        depth = 0
    if path1[depth].max_occurs == 0:
        return True
    univocal1 = univocal2 = True
    if path1[depth].model == 'sequence':
        idx1 = path1[depth].index(path1[depth + 1])
        idx2 = path2[depth].index(path2[depth + 1])
        before1 = any((not e.is_emptiable() for e in path1[depth][:idx1]))
        after1 = before2 = any((not e.is_emptiable() for e in path1[depth][idx1 + 1:idx2]))
        after2 = any((not e.is_emptiable() for e in path1[depth][idx2 + 1:]))
    else:
        before1 = after1 = before2 = after2 = False
    for k in range(depth + 1, len(path1) - 1):
        univocal1 &= path1[k].is_univocal()
        idx = path1[k].index(path1[k + 1])
        if path1[k].model == 'sequence':
            before1 |= any((not e.is_emptiable() for e in path1[k][:idx]))
            after1 |= any((not e.is_emptiable() for e in path1[k][idx + 1:]))
        elif any((e.is_emptiable() for e in path1[k] if e is not path1[k][idx])):
            univocal1 = False
    for k in range(depth + 1, len(path2) - 1):
        univocal2 &= path2[k].is_univocal()
        idx = path2[k].index(path2[k + 1])
        if path2[k].model == 'sequence':
            before2 |= any((not e.is_emptiable() for e in path2[k][:idx]))
            after2 |= any((not e.is_emptiable() for e in path2[k][idx + 1:]))
        elif any((e.is_emptiable() for e in path2[k] if e is not path2[k][idx])):
            univocal2 = False
    if path1[depth].model != 'sequence':
        if before1 and before2:
            return True
        elif before1:
            return univocal1 and path1[-1].is_univocal() or after1 or path1[depth].max_occurs == 1
        elif before2:
            return univocal2 and path2[-1].is_univocal() or after2 or path2[depth].max_occurs == 1
        else:
            return False
    elif path1[depth].max_occurs == 1:
        return before2 or ((before1 or univocal1) and (path1[-1].is_univocal() or after1))
    else:
        return (before2 or ((before1 or univocal1) and (path1[-1].is_univocal() or after1))) and (before1 or ((before2 or univocal2) and (path2[-1].is_univocal() or after2)))