from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def get_chunksize(client):
    """
    Default chunksize is 64MB
    """
    chunksize = None
    result = client['config'].settings.find_one({'_id': 'chunksize'})
    if not result:
        chunksize = 64
    else:
        chunksize = result['value']
    return chunksize