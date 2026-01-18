import functools
from openstack.cloud import _utils
from openstack.config import loader
from openstack import connection
from openstack import exceptions
def list_hosts(self, expand=True, fail_on_cloud_config=True, all_projects=False):
    hostvars = []
    for cloud in self.clouds:
        try:
            for server in cloud.list_servers(detailed=expand, all_projects=all_projects):
                hostvars.append(server)
        except exceptions.SDKException:
            if fail_on_cloud_config:
                raise
    return hostvars