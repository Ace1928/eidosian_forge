from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def remove_zone_range(client, namespace, min, max):
    """
    Remove a zone range.
    We do this by setting the zone to None
    @client - MongoDB connection
    @namespace - In the form database.collection
    @min - The min range value
    @max - The max range value
    """
    cmd_doc = OrderedDict([('updateZoneKeyRange', namespace), ('min', min), ('max', max), ('zone', None)])
    client['admin'].command(cmd_doc)