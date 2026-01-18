import collections.abc
import functools
import itertools
import threading
import typing as ty
import uuid
import warnings
import debtcollector
from debtcollector import renames
def is_user_context(context: RequestContext) -> bool:
    """Indicates if the request context is a normal user."""
    if not context or not isinstance(context, RequestContext):
        return False
    if context.is_admin:
        return False
    return True