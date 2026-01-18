import configparser
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import policy
import glance.api.policy
from glance.common import exception
from glance.i18n import _, _LE, _LW
def is_property_protection_enabled():
    if CONF.property_protection_file:
        return True
    return False