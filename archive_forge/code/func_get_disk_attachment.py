from __future__ import (absolute_import, division, print_function)
import traceback
import os
import ssl
import time
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def get_disk_attachment(disk, disk_attachments, connection):
    for disk_attachment in disk_attachments:
        if get_link_name(connection, disk_attachment.disk) == disk.get('name') or disk_attachment.disk.id == disk.get('id'):
            return disk_attachment