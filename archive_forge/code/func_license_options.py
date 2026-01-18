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
@property
def license_options(self):
    result = dict(eula=self.eula or '', email=self.email or '', first_name=self.first_name or '', last_name=self.last_name or '', company=self.company or '', phone=self.phone or '', job_title=self.job_title or '', address=self.address or '', city=self.city or '', state=self.state or '', postal_code=self.postal_code or '', country=self.country or '')
    return result