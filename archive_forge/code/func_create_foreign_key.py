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
def create_foreign_key(self, constraint_name: str, referent_table: str, local_cols: List[str], remote_cols: List[str], *, referent_schema: Optional[str]=None, onupdate: Optional[str]=None, ondelete: Optional[str]=None, deferrable: Optional[bool]=None, initially: Optional[str]=None, match: Optional[str]=None, **dialect_kw: Any) -> None:
    """Issue a "create foreign key" instruction using the
            current batch migration context.

            The batch form of this call omits the ``source`` and ``source_schema``
            arguments from the call.

            e.g.::

                with batch_alter_table("address") as batch_op:
                    batch_op.create_foreign_key(
                        "fk_user_address",
                        "user",
                        ["user_id"],
                        ["id"],
                    )

            .. seealso::

                :meth:`.Operations.create_foreign_key`

            """
    ...