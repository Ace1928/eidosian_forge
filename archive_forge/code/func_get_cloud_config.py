import os
import os_client_config
from tempest.lib.cli import base
def get_cloud_config(cloud='devstack-admin'):
    return os_client_config.OpenStackConfig().get_one_cloud(cloud=cloud)