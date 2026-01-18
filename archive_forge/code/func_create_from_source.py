from openstack import exceptions
from openstack import resource
from openstack import utils
@classmethod
def create_from_source(cls, session, group_snapshot_id, source_group_id, name=None, description=None):
    """Creates a new group from source."""
    session = cls._get_session(session)
    microversion = cls._get_microversion(session, action='create')
    url = utils.urljoin(cls.base_path, 'action')
    body = {'create-from-src': {'name': name, 'description': description, 'group_snapshot_id': group_snapshot_id, 'source_group_id': source_group_id}}
    response = session.post(url, json=body, microversion=microversion)
    exceptions.raise_from_response(response)
    group = Group()
    group._translate_response(response=response)
    return group