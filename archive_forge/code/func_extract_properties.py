import logging
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient import exceptions
def extract_properties(properties):
    result_dict = {}
    for item in properties:
        try:
            key, value = item.split('=', 1)
            if key in result_dict:
                raise exceptions.CommandError("Argument '%s' is specified twice." % key)
            else:
                result_dict[key] = value
        except ValueError:
            raise exceptions.CommandError("Parsing error, expected format 'key=value' for " + item)
    return result_dict