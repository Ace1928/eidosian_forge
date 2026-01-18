from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
@staticmethod
def sanitize_wwn(initiator):
    """ igroup initiator may or may not be using WWN format: eg 20:00:00:25:B5:00:20:01
            if format is matched, convert initiator to lowercase, as this is what ONTAP is using """
    wwn_format = '[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){7}'
    initiator = initiator.strip()
    if re.match(wwn_format, initiator):
        initiator = initiator.lower()
    return initiator