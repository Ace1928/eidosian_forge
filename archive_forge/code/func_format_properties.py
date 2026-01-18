import logging
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient import exceptions
def format_properties(properties):
    formatted_data = []
    for item in properties:
        formatted_data.append('%s : %s' % (item, properties[item]))
    return '\n'.join(formatted_data)