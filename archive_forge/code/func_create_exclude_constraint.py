from __future__ import annotations
from contextlib import contextmanager
import re
import textwrap
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List  # noqa
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence  # noqa
from typing import Tuple
from typing import Type  # noqa
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql.elements import conv
from . import batch
from . import schemaobj
from .. import util
from ..util import sqla_compat
from ..util.compat import formatannotation_fwdref
from ..util.compat import inspect_formatargspec
from ..util.compat import inspect_getfullargspec
from ..util.sqla_compat import _literal_bindparam
def create_exclude_constraint(self, constraint_name: str, *elements: Any, **kw: Any) -> Optional[Table]:
    """Issue a "create exclude constraint" instruction using the
            current batch migration context.

            .. note::  This method is Postgresql specific, and additionally
               requires at least SQLAlchemy 1.0.

            .. seealso::

                :meth:`.Operations.create_exclude_constraint`

            """
    ...