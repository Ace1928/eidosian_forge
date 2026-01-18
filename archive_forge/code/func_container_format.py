from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
@container_format.setter
def container_format(self, value):
    if hasattr(self, '_container_format') and self.status not in ('queued', 'importing'):
        msg = _('Attribute container_format can be only replaced for a queued image.')
        raise exception.Forbidden(message=msg)
    self._container_format = value