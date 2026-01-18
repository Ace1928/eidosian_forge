from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import quote
import json
import re
import xml.etree.ElementTree as ET
def attr_name(self, _id):
    """
        Get attribute name from hex ID
        :param _id: The hex ID to lookup a name for
        :type _id: str
        :returns: Translated name of hex ID, or None if no translation found
        :rtype: str or None
        """
    for name, m_id in list(self.attr_map.items()):
        if _id == m_id:
            return name
    return None