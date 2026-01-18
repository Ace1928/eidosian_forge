import logging
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient import exceptions
def format_share_group_type(share_group_type, formatter='table'):
    printable_share_group_type = share_group_type._info
    is_public = printable_share_group_type.pop('is_public')
    printable_share_group_type['visibility'] = 'public' if is_public else 'private'
    if formatter == 'table':
        printable_share_group_type['group_specs'] = format_properties(share_group_type.group_specs)
        printable_share_group_type['share_types'] = '\n'.join(printable_share_group_type['share_types'])
    return printable_share_group_type