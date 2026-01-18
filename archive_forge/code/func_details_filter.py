import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
@property
def details_filter(self):
    """Callback (may be none) to call to discard events + details."""
    return self._details_filter