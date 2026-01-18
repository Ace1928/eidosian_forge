from __future__ import absolute_import, division, print_function
import re
import uuid
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def templates_differ(self):
    backup = self.want.name
    self.want.update({'name': 'ansible-{0}'.format(str(uuid.uuid4()))})
    temp = self._get_temporary_template()
    self.want.update({'name': backup})
    if temp.checksum != self.have.checksum:
        return True
    return False