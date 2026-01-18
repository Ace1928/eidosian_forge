import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@payload_content_type.setter
@immutable_after_save
def payload_content_type(self, value):
    self._meta['payload_content_type'] = value