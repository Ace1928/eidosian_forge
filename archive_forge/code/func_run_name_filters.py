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
def run_name_filters(self, name: Optional[str], type_: NameFilterType, parent_names: NameFilterParentNames) -> bool:
    """Run the context's name filters and return True if the targets
        should be part of the autogenerate operation.

        This method should be run for every kind of name encountered within the
        reflection side of an autogenerate operation, giving the environment
        the chance to filter what names should be reflected as database
        objects.  The filters here are produced directly via the
        :paramref:`.EnvironmentContext.configure.include_name` parameter.

        """
    if 'schema_name' in parent_names:
        if type_ == 'table':
            table_name = name
        else:
            table_name = parent_names.get('table_name', None)
        if table_name:
            schema_name = parent_names['schema_name']
            if schema_name:
                parent_names['schema_qualified_table_name'] = '%s.%s' % (schema_name, table_name)
            else:
                parent_names['schema_qualified_table_name'] = table_name
    for fn in self._name_filters:
        if not fn(name, type_, parent_names):
            return False
    else:
        return True