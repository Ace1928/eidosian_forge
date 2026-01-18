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
def get_default_redirect(self, rule: Rule, method: str, values: t.MutableMapping[str, t.Any], query_args: t.Mapping[str, t.Any] | str) -> str | None:
    """A helper that returns the URL to redirect to if it finds one.
        This is used for default redirecting only.

        :internal:
        """
    assert self.map.redirect_defaults
    for r in self.map._rules_by_endpoint[rule.endpoint]:
        if r is rule:
            break
        if r.provides_defaults_for(rule) and r.suitable_for(values, method):
            values.update(r.defaults)
            domain_part, path = r.build(values)
            return self.make_redirect_url(path, query_args, domain_part=domain_part)
    return None