from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def is_version_with_default_network(self):
    """Is current BIG-IP version missing "default" network value support

        Returns:
            bool: True when it is missing. False otherwise.
        """
    version = tmos_version(self.client)
    if Version(version) < Version('13.1.0'):
        return True
    else:
        return False