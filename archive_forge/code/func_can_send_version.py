import abc
import logging
from oslo_config import cfg
from oslo_messaging._drivers import base as driver_base
from oslo_messaging import _metrics as metrics
from oslo_messaging import _utils as utils
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import transport as msg_transport
def can_send_version(self, version=_marker):
    """Check to see if a version is compatible with the version cap."""
    return self.prepare(version=version).can_send_version()