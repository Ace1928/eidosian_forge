import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def immutable_after_save(func):

    @functools.wraps(func)
    def wrapper(self, *args):
        if self._order_ref:
            raise base.ImmutableException()
        return func(self, *args)
    return wrapper