from __future__ import annotations
import collections.abc as collections_abc
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.parse import unquote
from .interfaces import Dialect
from .. import exc
from .. import util
from ..dialects import plugins
from ..dialects import registry
def get_driver_name(self) -> str:
    """Return the backend name.

        This is the name that corresponds to the DBAPI driver in
        use, and is the portion of the :attr:`_engine.URL.drivername`
        that is to the right of the plus sign.

        If the :attr:`_engine.URL.drivername` does not include a plus sign,
        then the default :class:`_engine.Dialect` for this :class:`_engine.URL`
        is imported in order to get the driver name.

        """
    if '+' not in self.drivername:
        return self.get_dialect().driver
    else:
        return self.drivername.split('+')[1]