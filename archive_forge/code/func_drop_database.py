from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.influxdb import InfluxDb
def drop_database(module, client, database_name):
    if not module.check_mode:
        try:
            client.drop_database(database_name)
        except exceptions.InfluxDBClientError as e:
            module.fail_json(msg=e.content)
    module.exit_json(changed=True)