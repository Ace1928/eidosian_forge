import collections.abc
import functools
import itertools
import threading
import typing as ty
import uuid
import warnings
import debtcollector
from debtcollector import renames
@property
def global_id(self) -> str:
    """Return a sensible value for global_id to pass on.

        When we want to make a call with to another service, it's
        important that we try to use global_request_id if available,
        and fall back to the locally generated request_id if not.
        """
    return self.global_request_id or self.request_id