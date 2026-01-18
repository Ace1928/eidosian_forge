import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@property
@lazy
def plugin_ca_id(self):
    return self._plugin_ca_id