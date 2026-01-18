from oslo_serialization import jsonutils
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib.db import constants as db_const
from neutron_lib import exceptions
def convert_az_list_to_string(az_list):
    """Convert a list of availability zones into a string.

    :param az_list: A list of AZs.
    :returns: The az_list in string format.
    """
    return jsonutils.dumps(az_list)