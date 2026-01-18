from __future__ import (absolute_import, division, print_function)
import logging
import time
from ansible.module_utils.basic import missing_required_lib
def format_client_id(self, client_id):
    return client_id if client_id.endswith('clients') else client_id + 'clients'