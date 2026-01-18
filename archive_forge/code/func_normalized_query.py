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
@property
def normalized_query(self) -> Mapping[str, Sequence[str]]:
    """Return the :attr:`_engine.URL.query` dictionary with values normalized
        into sequences.

        As the :attr:`_engine.URL.query` dictionary may contain either
        string values or sequences of string values to differentiate between
        parameters that are specified multiple times in the query string,
        code that needs to handle multiple parameters generically will wish
        to use this attribute so that all parameters present are presented
        as sequences.   Inspiration is from Python's ``urllib.parse.parse_qs``
        function.  E.g.::


            >>> from sqlalchemy.engine import make_url
            >>> url = make_url("postgresql+psycopg2://user:pass@host/dbname?alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt")
            >>> url.query
            immutabledict({'alt_host': ('host1', 'host2'), 'ssl_cipher': '/path/to/crt'})
            >>> url.normalized_query
            immutabledict({'alt_host': ('host1', 'host2'), 'ssl_cipher': ('/path/to/crt',)})

        """
    return util.immutabledict({k: (v,) if not isinstance(v, tuple) else v for k, v in self.query.items()})