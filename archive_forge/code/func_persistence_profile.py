from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def persistence_profile(self):
    """Return persistence profile in a consumable form

        I don't know why the persistence profile is stored this way, but below is the
        general format of it.

            "persist": [
                {
                    "name": "msrdp",
                    "partition": "Common",
                    "tmDefault": "yes",
                    "nameReference": {
                        "link": "https://localhost/mgmt/tm/ltm/persistence/msrdp/~Common~msrdp?ver=13.1.0.4"
                    }
                }
            ],

        As you can see, this is quite different from something like the fallback
        persistence profile which is just simply

            /Common/fallback1

        This method makes the persistence profile look like the fallback profile.

        Returns:
             string: The persistence profile configured on the virtual.
        """
    if self._values['persistence_profile'] is None:
        return None
    profile = self._values['persistence_profile'][0]
    result = fq_name(profile['partition'], profile['name'])
    return result