from __future__ import annotations
import re
from typing import Any
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import cast
from sqlalchemy import JSON
from sqlalchemy import schema
from sqlalchemy import sql
from .base import alter_table
from .base import format_table_name
from .base import RenameTable
from .impl import DefaultImpl
from .. import util
from ..util.sqla_compat import compiles
def cast_for_batch_migrate(self, existing: Column[Any], existing_transfer: Dict[str, Union[TypeEngine, Cast]], new_type: TypeEngine) -> None:
    if existing.type._type_affinity is not new_type._type_affinity and (not isinstance(new_type, JSON)):
        existing_transfer['expr'] = cast(existing_transfer['expr'], new_type)