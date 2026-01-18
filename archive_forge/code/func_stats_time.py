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
@stats_time.setter
def stats_time(self, stats_time):
    get_validator(ConfWithStats.STATS_TIME)(stats_time)
    if stats_time != self.stats_time:
        self._settings[ConfWithStats.STATS_TIME] = stats_time
        self._notify_listeners(ConfWithStats.UPDATE_STATS_TIME_EVT, stats_time)