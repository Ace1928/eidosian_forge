from __future__ import annotations
from collections import deque
from enum import Enum
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util import langhelpers
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
def cloned_traverse(obj: Optional[ExternallyTraversible], opts: Mapping[str, Any], visitors: Mapping[str, _TraverseCallableType[Any]]) -> Optional[ExternallyTraversible]:
    """Clone the given expression structure, allowing modifications by
    visitors for mutable objects.

    Traversal usage is the same as that of :func:`.visitors.traverse`.
    The visitor functions present in the ``visitors`` dictionary may also
    modify the internals of the given structure as the traversal proceeds.

    The :func:`.cloned_traverse` function does **not** provide objects that are
    part of the :class:`.Immutable` interface to the visit methods (this
    primarily includes :class:`.ColumnClause`, :class:`.Column`,
    :class:`.TableClause` and :class:`.Table` objects). As this traversal is
    only intended to allow in-place mutation of objects, :class:`.Immutable`
    objects are skipped. The :meth:`.Immutable._clone` method is still called
    on each object to allow for objects to replace themselves with a different
    object based on a clone of their sub-internals (e.g. a
    :class:`.ColumnClause` that clones its subquery to return a new
    :class:`.ColumnClause`).

    .. versionchanged:: 2.0  The :func:`.cloned_traverse` function omits
       objects that are part of the :class:`.Immutable` interface.

    The central API feature used by the :func:`.visitors.cloned_traverse`
    and :func:`.visitors.replacement_traverse` functions, in addition to the
    :meth:`_expression.ClauseElement.get_children`
    function that is used to achieve
    the iteration, is the :meth:`_expression.ClauseElement._copy_internals`
    method.
    For a :class:`_expression.ClauseElement`
    structure to support cloning and replacement
    traversals correctly, it needs to be able to pass a cloning function into
    its internal members in order to make copies of them.

    .. seealso::

        :func:`.visitors.traverse`

        :func:`.visitors.replacement_traverse`

    """
    cloned: Dict[int, ExternallyTraversible] = {}
    stop_on = set(opts.get('stop_on', []))

    def deferred_copy_internals(obj: ExternallyTraversible) -> ExternallyTraversible:
        return cloned_traverse(obj, opts, visitors)

    def clone(elem: ExternallyTraversible, **kw: Any) -> ExternallyTraversible:
        if elem in stop_on:
            return elem
        else:
            if id(elem) not in cloned:
                if 'replace' in kw:
                    newelem = cast(Optional[ExternallyTraversible], kw['replace'](elem))
                    if newelem is not None:
                        cloned[id(elem)] = newelem
                        return newelem
                cloned[id(elem)] = newelem = elem._clone(clone=clone, **kw)
                newelem._copy_internals(clone=clone, **kw)
                if not elem._is_immutable:
                    meth = visitors.get(newelem.__visit_name__, None)
                    if meth:
                        meth(newelem)
            return cloned[id(elem)]
    if obj is not None:
        obj = clone(obj, deferred_copy_internals=deferred_copy_internals, **opts)
    clone = None
    return obj