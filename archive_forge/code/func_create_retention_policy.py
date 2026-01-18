from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.influxdb import InfluxDb
from ansible.module_utils.common.text.converters import to_native
def create_retention_policy(module, client):
    database_name = module.params['database_name']
    policy_name = module.params['policy_name']
    duration = module.params['duration']
    replication = module.params['replication']
    default = module.params['default']
    shard_group_duration = module.params['shard_group_duration']
    if not check_duration_literal(duration):
        module.fail_json(msg='Failed to parse value of duration')
    influxdb_duration_format = parse_duration_literal(duration)
    if influxdb_duration_format != 0 and influxdb_duration_format < MINIMUM_VALID_DURATION:
        module.fail_json(msg='duration value must be at least 1h')
    if shard_group_duration is not None:
        if not check_duration_literal(shard_group_duration):
            module.fail_json(msg='Failed to parse value of shard_group_duration')
        influxdb_shard_group_duration_format = parse_duration_literal(shard_group_duration)
        if influxdb_shard_group_duration_format < MINIMUM_VALID_SHARD_GROUP_DURATION:
            module.fail_json(msg='shard_group_duration value must be finite and at least 1h')
    if not module.check_mode:
        try:
            if shard_group_duration:
                client.create_retention_policy(policy_name, duration, replication, database_name, default, shard_group_duration)
            else:
                client.create_retention_policy(policy_name, duration, replication, database_name, default)
        except exceptions.InfluxDBClientError as e:
            module.fail_json(msg=e.content)
    module.exit_json(changed=True)