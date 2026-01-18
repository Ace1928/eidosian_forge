from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.influxdb import InfluxDb
from ansible.module_utils.common.text.converters import to_native
def drop_retention_policy(module, client):
    database_name = module.params['database_name']
    policy_name = module.params['policy_name']
    if not module.check_mode:
        try:
            client.drop_retention_policy(policy_name, database_name)
        except exceptions.InfluxDBClientError as e:
            module.fail_json(msg=e.content)
    module.exit_json(changed=True)