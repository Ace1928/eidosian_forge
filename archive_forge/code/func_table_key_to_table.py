from __future__ import annotations
import contextlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import inspect
from . import compare
from . import render
from .. import util
from ..operations import ops
from ..util import sqla_compat
@util.memoized_property
def table_key_to_table(self) -> Dict[str, Table]:
    """Return an aggregate  of the :attr:`.MetaData.tables` dictionaries.

        The :attr:`.MetaData.tables` collection is a dictionary of table key
        to :class:`.Table`; this method aggregates the dictionary across
        multiple :class:`.MetaData` objects into one dictionary.

        Duplicate table keys are **not** supported; if two :class:`.MetaData`
        objects contain the same table key, an exception is raised.

        """
    result: Dict[str, Table] = {}
    for m in util.to_list(self.metadata):
        intersect = set(result).intersection(set(m.tables))
        if intersect:
            raise ValueError('Duplicate table keys across multiple MetaData objects: %s' % ', '.join(('"%s"' % key for key in sorted(intersect))))
        result.update(m.tables)
    return result