from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import utils
def create_from_src(self, cgsnapshot_id, source_cgid, name=None, description=None, user_id=None, project_id=None):
    """Creates a consistency group from a cgsnapshot or a source CG.

        :param cgsnapshot_id: UUID of a CGSnapshot
        :param source_cgid: UUID of a source CG
        :param name: Name of the ConsistencyGroup
        :param description: Description of the ConsistencyGroup
        :param user_id: User id derived from context
        :param project_id: Project id derived from context
        :rtype: A dictionary containing Consistencygroup metadata
        """
    body = {'consistencygroup-from-src': {'name': name, 'description': description, 'cgsnapshot_id': cgsnapshot_id, 'source_cgid': source_cgid, 'user_id': user_id, 'project_id': project_id, 'status': 'creating'}}
    self.run_hooks('modify_body_for_update', body, 'consistencygroup-from-src')
    resp, body = self.api.client.post('/consistencygroups/create_from_src', body=body)
    return common_base.DictWithMeta(body['consistencygroup'], resp)