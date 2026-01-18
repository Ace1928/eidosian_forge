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
def inline_literal(self, value: Union[str, int], type_: Optional[TypeEngine[Any]]=None) -> _literal_bindparam:
    """Produce an 'inline literal' expression, suitable for
        using in an INSERT, UPDATE, or DELETE statement.

        When using Alembic in "offline" mode, CRUD operations
        aren't compatible with SQLAlchemy's default behavior surrounding
        literal values,
        which is that they are converted into bound values and passed
        separately into the ``execute()`` method of the DBAPI cursor.
        An offline SQL
        script needs to have these rendered inline.  While it should
        always be noted that inline literal values are an **enormous**
        security hole in an application that handles untrusted input,
        a schema migration is not run in this context, so
        literals are safe to render inline, with the caveat that
        advanced types like dates may not be supported directly
        by SQLAlchemy.

        See :meth:`.Operations.execute` for an example usage of
        :meth:`.Operations.inline_literal`.

        The environment can also be configured to attempt to render
        "literal" values inline automatically, for those simple types
        that are supported by the dialect; see
        :paramref:`.EnvironmentContext.configure.literal_binds` for this
        more recently added feature.

        :param value: The value to render.  Strings, integers, and simple
         numerics should be supported.   Other types like boolean,
         dates, etc. may or may not be supported yet by various
         backends.
        :param type\\_: optional - a :class:`sqlalchemy.types.TypeEngine`
         subclass stating the type of this value.  In SQLAlchemy
         expressions, this is usually derived automatically
         from the Python type of the value itself, as well as
         based on the context in which the value is used.

        .. seealso::

            :paramref:`.EnvironmentContext.configure.literal_binds`

        """
    return sqla_compat._literal_bindparam(None, value, type_=type_)