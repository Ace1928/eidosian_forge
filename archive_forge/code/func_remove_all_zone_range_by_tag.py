from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def remove_all_zone_range_by_tag(client, tag):
    result = client['config'].tags.find({'tag': tag})
    for r in result:
        remove_zone_range(client, r['ns'], r['min'], r['max'])