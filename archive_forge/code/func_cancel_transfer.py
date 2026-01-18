from __future__ import (absolute_import, division, print_function)
import json
import os
import subprocess
import time
import traceback
import inspect
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def cancel_transfer(connection, transfer_id):
    transfer_service = connection.system_service().image_transfers_service().image_transfer_service(transfer_id)
    transfer_service.cancel()