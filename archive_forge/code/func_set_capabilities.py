import logging
import threading
import enum
from oslo_utils import reflection
from glance_store import exceptions
from glance_store.i18n import _LW
def set_capabilities(self, *dynamic_capabilites):
    """
        Set dynamic storage capabilities based on current
        driver configuration and backend status.

        :param dynamic_capabilites: dynamic storage capability(s).
        """
    for cap in dynamic_capabilites:
        self._capabilities |= int(cap)