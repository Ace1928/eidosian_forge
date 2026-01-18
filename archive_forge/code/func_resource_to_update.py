from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest
import json
import time
def resource_to_update(module):
    instance = resource_to_request(module)
    instance['name'] = 'projects/{0}/instances/{1}'.format(module.params['project'], module.params['name'])
    instance['config'] = 'projects/{0}/instanceConfigs/{1}'.format(module.params['project'], instance['config'])
    return {'instance': instance, 'fieldMask': "'name' ,'config' ,'displayName' ,'nodeCount' ,'processingUnits' ,'labels'"}