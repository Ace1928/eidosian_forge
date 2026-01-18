import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@property
def ca_ref(self):
    return self._ca_ref