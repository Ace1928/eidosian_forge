from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_subvolume_children(self, subvolume_id):
    return [BtrfsSubvolume(self, x['id']) for x in self.__subvolumes.values() if x['parent'] == subvolume_id]