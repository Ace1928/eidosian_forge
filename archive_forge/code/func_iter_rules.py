from __future__ import annotations
import typing as t
import warnings
from pprint import pformat
from threading import Lock
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlunsplit
from .._internal import _get_environ
from .._internal import _wsgi_decoding_dance
from ..datastructures import ImmutableDict
from ..datastructures import MultiDict
from ..exceptions import BadHost
from ..exceptions import HTTPException
from ..exceptions import MethodNotAllowed
from ..exceptions import NotFound
from ..urls import _urlencode
from ..wsgi import get_host
from .converters import DEFAULT_CONVERTERS
from .exceptions import BuildError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .exceptions import RequestRedirect
from .exceptions import WebsocketMismatch
from .matcher import StateMachineMatcher
from .rules import _simple_rule_re
from .rules import Rule
def iter_rules(self, endpoint: str | None=None) -> t.Iterator[Rule]:
    """Iterate over all rules or the rules of an endpoint.

        :param endpoint: if provided only the rules for that endpoint
                         are returned.
        :return: an iterator
        """
    self.update()
    if endpoint is not None:
        return iter(self._rules_by_endpoint[endpoint])
    return iter(self._rules)