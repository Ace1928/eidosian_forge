from __future__ import annotations
import contextlib
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import event
from sqlalchemy import inspect
from sqlalchemy import schema as sa_schema
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.sql import expression
from sqlalchemy.sql.schema import ForeignKeyConstraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.sql.schema import UniqueConstraint
from sqlalchemy.util import OrderedSet
from .. import util
from ..ddl._autogen import is_index_sig
from ..ddl._autogen import is_uq_sig
from ..operations import ops
from ..util import sqla_compat
def obj_removed(obj: _constraint_sig):
    if is_index_sig(obj):
        if obj.is_unique and (not supports_unique_constraints):
            return
        if autogen_context.run_object_filters(obj.const, obj.name, 'index', True, None):
            modify_ops.ops.append(ops.DropIndexOp.from_index(obj.const))
            log.info('Detected removed index %r on %r', obj.name, tname)
    elif is_uq_sig(obj):
        if is_create_table or is_drop_table:
            return
        if autogen_context.run_object_filters(obj.const, obj.name, 'unique_constraint', True, None):
            modify_ops.ops.append(ops.DropConstraintOp.from_constraint(obj.const))
            log.info('Detected removed unique constraint %r on %r', obj.name, tname)
    else:
        assert False