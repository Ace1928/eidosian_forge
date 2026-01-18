from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import utils
@api_versions.wraps('3.38')
def list_replication_targets(self, group):
    """List replication targets for a group.

        :param group: the :class:`Group` to list replication targets.
        """
    body = {'list_replication_targets': {}}
    self.run_hooks('modify_body_for_action', body, 'group')
    url = '/groups/%s/action' % base.getid(group)
    resp, body = self.api.client.post(url, body=body)
    return common_base.TupleWithMeta((resp, body), resp)