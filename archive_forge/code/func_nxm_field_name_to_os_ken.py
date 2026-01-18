import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext
def nxm_field_name_to_os_ken(field):
    """
    Convert an ovs-ofctl style NXM_/OXM_ field name to
    a os_ken match field name.
    """
    if field.endswith('_W'):
        field = field[:-2]
    prefix = field[:7]
    field = field[7:].lower()
    mapped_result = None
    if prefix == 'NXM_NX_':
        mapped_result = _NXM_FIELD_MAP.get(field)
    elif prefix == 'NXM_OF_':
        mapped_result = _NXM_OF_FIELD_MAP.get(field)
    elif prefix == 'OXM_OF_':
        pass
    else:
        raise ValueError
    if mapped_result is not None:
        return mapped_result
    return field