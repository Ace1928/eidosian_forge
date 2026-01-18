import itertools
import logging
import operator
from oslo_messaging import dispatcher
from oslo_messaging import serializer as msg_serializer
@property
def supported_priorities(self):
    return self._callbacks_by_priority.keys()