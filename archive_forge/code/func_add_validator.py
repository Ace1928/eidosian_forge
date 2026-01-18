import collections
import functools
import inspect
import re
import netaddr
from os_ken.lib.packet import ether_types
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from webob import exc
from neutron_lib._i18n import _
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.plugins import directory
from neutron_lib.services.qos import constants as qos_consts
def add_validator(validation_type, validator):
    """Dynamically add a validator.

    This can be used by clients to add their own, private validators, rather
    than directly modifying the data structure. The clients can NOT modify
    existing validators.
    """
    key = _to_validation_type(validation_type)
    if key in validators:
        if inspect.getsource(validator) != inspect.getsource(validators[key]):
            msg = _('Validator type %s is already defined') % validation_type
            raise KeyError(msg)
        return
    validators[key] = validator