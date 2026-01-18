import abc
import contextlib
import functools
import itertools
import threading
from oslo_utils import excutils
from oslo_utils import timeutils
from taskflow.conductors import base
from taskflow import exceptions as excp
from taskflow.listeners import logging as logging_listener
from taskflow import logging
from taskflow import states
from taskflow.types import timing as tt
from taskflow.utils import iter_utils
from taskflow.utils import misc
@property
def dispatching(self):
    """Whether or not the dispatching loop is still dispatching."""
    return not self._dead.is_set()