from __future__ import annotations
import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self
@property
def returned_defaults(self):
    """Return the values of default columns that were fetched using
        the :meth:`.ValuesBase.return_defaults` feature.

        The value is an instance of :class:`.Row`, or ``None``
        if :meth:`.ValuesBase.return_defaults` was not used or if the
        backend does not support RETURNING.

        .. seealso::

            :meth:`.ValuesBase.return_defaults`

        """
    if self.context.executemany:
        raise exc.InvalidRequestError('This statement was an executemany call; if return defaults is supported, please use .returned_defaults_rows.')
    rows = self.context.returned_default_rows
    if rows:
        return rows[0]
    else:
        return None