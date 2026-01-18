import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@property
def order_ref(self):
    return self._order_ref