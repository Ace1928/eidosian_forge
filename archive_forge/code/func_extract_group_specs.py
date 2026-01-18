import logging
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient import exceptions
def extract_group_specs(extra_specs, specs_to_add):
    return extract_extra_specs(extra_specs, specs_to_add, constants.GROUP_BOOL_SPECS)