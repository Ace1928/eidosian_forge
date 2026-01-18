from __future__ import absolute_import, division, print_function
import re
import time
import xml.etree.ElementTree
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import iControlRestSession
from ..module_utils.teem import send_teem
def set_result_for_license_fault(self, result):
    root = self.find_element('LicensingFault')
    if root is None:
        return result
    for elem in root:
        if elem.tag == 'faultNumber':
            result['faultNumber'] = elem.text
        elif elem.tag == 'faultText':
            tmp = elem.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}nil', None)
            if tmp == 'true':
                result['faultText'] = None
            else:
                result['faultText'] = elem.text