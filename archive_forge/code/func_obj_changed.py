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
def obj_changed(old: _constraint_sig, new: _constraint_sig, msg: str):
    if is_index_sig(old):
        assert is_index_sig(new)
        if autogen_context.run_object_filters(new.const, new.name, 'index', False, old.const):
            log.info('Detected changed index %r on %r: %s', old.name, tname, msg)
            modify_ops.ops.append(ops.DropIndexOp.from_index(old.const))
            modify_ops.ops.append(ops.CreateIndexOp.from_index(new.const))
    elif is_uq_sig(old):
        assert is_uq_sig(new)
        if autogen_context.run_object_filters(new.const, new.name, 'unique_constraint', False, old.const):
            log.info('Detected changed unique constraint %r on %r: %s', old.name, tname, msg)
            modify_ops.ops.append(ops.DropConstraintOp.from_constraint(old.const))
            modify_ops.ops.append(ops.AddConstraintOp.from_constraint(new.const))
    else:
        assert False