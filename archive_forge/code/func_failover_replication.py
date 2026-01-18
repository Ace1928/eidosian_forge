from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import utils
@api_versions.wraps('3.38')
def failover_replication(self, group, allow_attached_volume=False, secondary_backend_id=None):
    """fails over replication for a group.

        :param group: the :class:`Group` to failover.
        :param allow attached volumes: allow attached volumes in the group.
        :param secondary_backend_id: secondary backend id.
        """
    body = {'failover_replication': {'allow_attached_volume': allow_attached_volume, 'secondary_backend_id': secondary_backend_id}}
    self.run_hooks('modify_body_for_action', body, 'group')
    url = '/groups/%s/action' % base.getid(group)
    resp, body = self.api.client.post(url, body=body)
    return common_base.TupleWithMeta((resp, body), resp)