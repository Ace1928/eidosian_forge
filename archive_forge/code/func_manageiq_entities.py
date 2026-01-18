from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def manageiq_entities():
    return {'provider': 'providers', 'host': 'hosts', 'vm': 'vms', 'category': 'categories', 'cluster': 'clusters', 'data store': 'data_stores', 'group': 'groups', 'resource pool': 'resource_pools', 'service': 'services', 'service template': 'service_templates', 'template': 'templates', 'tenant': 'tenants', 'user': 'users', 'blueprint': 'blueprints'}