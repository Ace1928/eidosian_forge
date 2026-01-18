from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def is_version_v1(self):
    """Checks to see if the TMOS version is less than 12.1.0

        Versions prior to 12.1.0 have a bug which prevents the REST
        API from properly listing any UCS files when you query the
        /mgmt/tm/sys/ucs endpoint. Therefore you need to do everything
        through tmsh over REST.

        :return: bool
        """
    version = tmos_version(self.client)
    if Version(version) < Version('12.1.0'):
        return True
    else:
        return False