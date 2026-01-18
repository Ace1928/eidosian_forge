from __future__ import absolute_import, division, print_function
import os
import re
import traceback
import shutil
import tempfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def db_weight(self):
    """Report the weight of the database.

        Normally, just 1, but for replication this is 0, and for 'all', this is more than 2.
        """
    if self['db'] == 'all':
        return 100000
    if self['db'] == 'replication':
        return 0
    if self['db'] in ['samerole', 'samegroup']:
        return 1
    return 1 + self['db'].count(',')