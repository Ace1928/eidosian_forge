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
def create_primary_key(self, constraint_name: str, columns: List[str]) -> None:
    """Issue a "create primary key" instruction using the
            current batch migration context.

            The batch form of this call omits the ``table_name`` and ``schema``
            arguments from the call.

            .. seealso::

                :meth:`.Operations.create_primary_key`

            """
    ...