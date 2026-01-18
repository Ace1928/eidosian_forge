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
def get_validator(validation_type, default=None):
    """Get a registered validator by type.

    :param validation_type: The type to retrieve the validator for.
    :param default: A default value to return if the validator is
        not registered.
    :return: The validator if registered, otherwise the default value.
    """
    return validators.get(_to_validation_type(validation_type), default)