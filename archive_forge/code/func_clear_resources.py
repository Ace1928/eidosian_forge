import re
import traceback
from oslo_log import log
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import client
from manilaclient.tests.functional import utils
@classmethod
def clear_resources(cls, resources=None):
    """Deletes resources, that were created in test suites.

        This method tries to remove resources from resource list,
        if it is not found, assume it was deleted in test itself.
        It is expected, that all resources were added as LIFO
        due to restriction of deletion resources, that are in the chain.
        :param resources: dict with keys 'type','id','client',
            'deletion_params' and 'deleted'. Optional 'deletion_params'
            contains additional data needed to delete some type of resources.
            Ex:
            params = {
                'type': 'share_network_subnet',
                'id': 'share-network-subnet-id',
                'client': None,
                'deletion_params': {
                    'share_network': 'share-network-id',
                },
                'deleted': False,
            }
        """
    if resources is None:
        resources = cls.method_resources
    for res in resources:
        if 'deleted' not in res:
            res['deleted'] = False
        if 'client' not in res:
            res['client'] = cls.get_cleanup_client()
        if not res['deleted']:
            res_id = res['id']
            client = res['client']
            deletion_params = res.get('deletion_params')
            with handle_cleanup_exceptions():
                if res['type'] == 'share_type':
                    client.delete_share_type(res_id, microversion=res['microversion'])
                    client.wait_for_share_type_deletion(res_id, microversion=res['microversion'])
                elif res['type'] == 'share_network':
                    client.delete_share_network(res_id, microversion=res['microversion'])
                    client.wait_for_share_network_deletion(res_id, microversion=res['microversion'])
                elif res['type'] == 'share_network_subnet':
                    client.delete_share_network_subnet(share_network_subnet=res_id, share_network=deletion_params['share_network'], microversion=res['microversion'])
                    client.wait_for_share_network_subnet_deletion(share_network_subnet=res_id, share_network=deletion_params['share_network'], microversion=res['microversion'])
                elif res['type'] == 'share':
                    client.delete_share(res_id, microversion=res['microversion'])
                    client.wait_for_share_deletion(res_id, microversion=res['microversion'])
                elif res['type'] == 'snapshot':
                    client.delete_snapshot(res_id, microversion=res['microversion'])
                    client.wait_for_snapshot_deletion(res_id, microversion=res['microversion'])
                elif res['type'] == 'share_replica':
                    client.delete_share_replica(res_id, microversion=res['microversion'])
                    client.wait_for_share_replica_deletion(res_id, microversion=res['microversion'])
                else:
                    LOG.warning("Provided unsupported resource type for cleanup '%s'. Skipping.", res['type'])
            res['deleted'] = True