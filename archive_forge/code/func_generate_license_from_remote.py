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
def generate_license_from_remote(self):
    mgmt = iControlRestSession(validate_certs=False, headers={'SOAPAction': '""', 'Content-Type': 'text/xml; charset=utf-8'})
    for x in range(0, 10):
        try:
            resp = mgmt.post(self.want.license_url, data=self.want.license_envelope)
        except Exception:
            raise
        try:
            resp = LicenseXmlParser(content=resp.content)
            result = resp.json()
        except F5ModuleError:
            raise
        except Exception:
            raise
        if result['state'] == 'EULA_REQUIRED':
            self.want.update({'eula': result['eula']})
            continue
        if result['state'] == 'LICENSE_RETURNED':
            return result
        elif result['state'] == 'EMAIL_REQUIRED':
            raise F5ModuleError('Email must be provided')
        elif result['state'] == 'CONTACT_INFO_REQUIRED':
            raise F5ModuleError('Contact info must be provided')
        else:
            raise F5ModuleError(result['fault_text'])