from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def remove_hal_links(self, records):
    """ Remove all _links entries """
    if isinstance(records, dict):
        records.pop('_links', None)
        for record in records.values():
            self.remove_hal_links(record)
    if isinstance(records, list):
        for record in records:
            self.remove_hal_links(record)