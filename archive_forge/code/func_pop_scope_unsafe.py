import copy
import sys
from contextlib import contextmanager
from sentry_sdk._compat import with_metaclass
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.scope import Scope
from sentry_sdk.client import Client
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._types import TYPE_CHECKING
def pop_scope_unsafe(self):
    """
        Pops a scope layer from the stack.

        Try to use the context manager :py:meth:`push_scope` instead.
        """
    rv = self._stack.pop()
    assert self._stack, 'stack must have at least one layer'
    return rv