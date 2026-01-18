from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def filter_pools(self, regexp='^$'):
    """
            Return a list of RhsmPools whose pool id matches the provided regular expression
        """
    r = re.compile(regexp)
    for product in self.products:
        if r.search(product.get_pool_id()):
            yield product