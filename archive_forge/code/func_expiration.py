import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@expiration.setter
@immutable_after_save
def expiration(self, value):
    self._meta['expiration'] = value