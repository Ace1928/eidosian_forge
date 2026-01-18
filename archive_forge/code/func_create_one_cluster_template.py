import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
@staticmethod
def create_one_cluster_template(attrs=None):
    """Create a fake ClusterTemplate.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with flavor_id, image_id, and so on
        """
    attrs = attrs or {}
    ct_info = {'links': [], 'insecure_registry': None, 'labels': None, 'updated_at': None, 'floating_ip_enabled': True, 'fixed_subnet': None, 'master_flavor_id': None, 'uuid': uuid.uuid4().hex, 'no_proxy': None, 'https_proxy': None, 'tls_disabled': False, 'keypair_id': None, 'public': False, 'http_proxy': None, 'docker_volume_size': None, 'server_type': 'vm', 'external_network_id': 'public', 'cluster_distro': 'fedora-atomic', 'image_id': 'fedora-atomic-latest', 'volume_driver': None, 'registry_enabled': False, 'docker_storage_driver': 'devicemapper', 'apiserver_port': None, 'name': 'fake-ct-' + uuid.uuid4().hex, 'created_at': datetime.datetime.now(), 'network_driver': 'flannel', 'fixed_network': None, 'coe': 'kubernetes', 'flavor_id': 'm1.medium', 'master_lb_enabled': False, 'dns_nameserver': '8.8.8.8', 'hidden': False, 'tags': ''}
    ct_info.update(attrs)
    ct = osc_fakes.FakeResource(info=copy.deepcopy(ct_info), loaded=True)
    return ct