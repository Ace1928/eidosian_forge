from openstack.common import metadata
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack import utils
@classmethod
def manage(cls, session, host, ref, name=None, description=None, volume_type=None, availability_zone=None, metadata=None, bootable=False, cluster=None):
    """Manage an existing volume."""
    url = '/manageable_volumes'
    if not utils.supports_microversion(session, '3.8'):
        url = '/os-volume-manage'
    body = {'volume': {'host': host, 'ref': ref, 'name': name, 'description': description, 'volume_type': volume_type, 'availability_zone': availability_zone, 'metadata': metadata, 'bootable': bootable}}
    if cluster is not None:
        body['volume']['cluster'] = cluster
    resp = session.post(url, json=body, microversion=cls._max_microversion)
    exceptions.raise_from_response(resp)
    volume = Volume()
    volume._translate_response(resp)
    return volume