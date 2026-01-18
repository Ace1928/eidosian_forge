import logging
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient import exceptions
def extract_key_value_options(pairs):
    result_dict = {}
    duplicate_options = []
    pairs = pairs or {}
    for attr, value in pairs.items():
        if attr not in result_dict:
            result_dict[attr] = value
        else:
            duplicate_options.append(attr)
    if pairs and len(duplicate_options) > 0:
        duplicate_str = ', '.join(duplicate_options)
        msg = 'Following options were duplicated: %s' % duplicate_str
        raise exceptions.CommandError(msg)
    return result_dict