import abc
from abc import ABCMeta
from abc import abstractmethod
import functools
import numbers
import logging
import uuid
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.base import RUNTIME_CONF_ERROR_CODE
from os_ken.services.protocols.bgp.base import validate
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
@stats_log_enabled.setter
def stats_log_enabled(self, enabled):
    get_validator(ConfWithStats.STATS_LOG_ENABLED)(enabled)
    if enabled != self.stats_log_enabled:
        self._settings[ConfWithStats.STATS_LOG_ENABLED] = enabled
        self._notify_listeners(ConfWithStats.UPDATE_STATS_LOG_ENABLED_EVT, enabled)