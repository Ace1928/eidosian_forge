from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def connect_to_influxdb(self):
    args = dict(host=self.hostname, port=self.port, username=self.username, password=self.password, database=self.database_name, ssl=self.params['ssl'], verify_ssl=self.params['validate_certs'], timeout=self.params['timeout'], use_udp=self.params['use_udp'], udp_port=self.params['udp_port'], proxies=self.params['proxies'])
    influxdb_api_version = LooseVersion(influxdb_version)
    if influxdb_api_version >= LooseVersion('4.1.0'):
        args.update(retries=self.params['retries'])
    if influxdb_api_version >= LooseVersion('5.1.0'):
        args.update(path=self.path)
    return InfluxDBClient(**args)