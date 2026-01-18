from __future__ import annotations
import contextlib
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence as typing_Sequence
from typing import Tuple
from . import roles
from .base import _generative
from .base import Executable
from .base import SchemaVisitor
from .elements import ClauseElement
from .. import exc
from .. import util
from ..util import topological
from ..util.typing import Protocol
from ..util.typing import Self
def sort_tables_and_constraints(tables, filter_fn=None, extra_dependencies=None, _warn_for_cycles=False):
    """Sort a collection of :class:`_schema.Table`  /
    :class:`_schema.ForeignKeyConstraint`
    objects.

    This is a dependency-ordered sort which will emit tuples of
    ``(Table, [ForeignKeyConstraint, ...])`` such that each
    :class:`_schema.Table` follows its dependent :class:`_schema.Table`
    objects.
    Remaining :class:`_schema.ForeignKeyConstraint`
    objects that are separate due to
    dependency rules not satisfied by the sort are emitted afterwards
    as ``(None, [ForeignKeyConstraint ...])``.

    Tables are dependent on another based on the presence of
    :class:`_schema.ForeignKeyConstraint` objects, explicit dependencies
    added by :meth:`_schema.Table.add_is_dependent_on`,
    as well as dependencies
    stated here using the :paramref:`~.sort_tables_and_constraints.skip_fn`
    and/or :paramref:`~.sort_tables_and_constraints.extra_dependencies`
    parameters.

    :param tables: a sequence of :class:`_schema.Table` objects.

    :param filter_fn: optional callable which will be passed a
     :class:`_schema.ForeignKeyConstraint` object,
     and returns a value based on
     whether this constraint should definitely be included or excluded as
     an inline constraint, or neither.   If it returns False, the constraint
     will definitely be included as a dependency that cannot be subject
     to ALTER; if True, it will **only** be included as an ALTER result at
     the end.   Returning None means the constraint is included in the
     table-based result unless it is detected as part of a dependency cycle.

    :param extra_dependencies: a sequence of 2-tuples of tables which will
     also be considered as dependent on each other.

    .. seealso::

        :func:`.sort_tables`


    """
    fixed_dependencies = set()
    mutable_dependencies = set()
    if extra_dependencies is not None:
        fixed_dependencies.update(extra_dependencies)
    remaining_fkcs = set()
    for table in tables:
        for fkc in table.foreign_key_constraints:
            if fkc.use_alter is True:
                remaining_fkcs.add(fkc)
                continue
            if filter_fn:
                filtered = filter_fn(fkc)
                if filtered is True:
                    remaining_fkcs.add(fkc)
                    continue
            dependent_on = fkc.referred_table
            if dependent_on is not table:
                mutable_dependencies.add((dependent_on, table))
        fixed_dependencies.update(((parent, table) for parent in table._extra_dependencies))
    try:
        candidate_sort = list(topological.sort(fixed_dependencies.union(mutable_dependencies), tables))
    except exc.CircularDependencyError as err:
        if _warn_for_cycles:
            util.warn('Cannot correctly sort tables; there are unresolvable cycles between tables "%s", which is usually caused by mutually dependent foreign key constraints.  Foreign key constraints involving these tables will not be considered; this warning may raise an error in a future release.' % (', '.join(sorted((t.fullname for t in err.cycles))),))
        for edge in err.edges:
            if edge in mutable_dependencies:
                table = edge[1]
                if table not in err.cycles:
                    continue
                can_remove = [fkc for fkc in table.foreign_key_constraints if filter_fn is None or filter_fn(fkc) is not False]
                remaining_fkcs.update(can_remove)
                for fkc in can_remove:
                    dependent_on = fkc.referred_table
                    if dependent_on is not table:
                        mutable_dependencies.discard((dependent_on, table))
        candidate_sort = list(topological.sort(fixed_dependencies.union(mutable_dependencies), tables))
    return [(table, table.foreign_key_constraints.difference(remaining_fkcs)) for table in candidate_sort] + [(None, list(remaining_fkcs))]