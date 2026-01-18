from manilaclient import api_versions
from manilaclient import base
@api_versions.wraps('2.57')
@api_versions.experimental_api
def migration_check(self, share_server, host, writable, nondisruptive, preserve_snapshots, new_share_network_id=None):
    """Check the share server migration to a new host

        :param share_server: either share_server object or text with its ID.
        :param host: Destination host where share server will be migrated.
        :param writable: Enforces migration to keep the shares writable.
        :param nondisruptive: Enforces migration to be nondisruptive.
        :param preserve_snapshots: Enforces migration to preserve snapshots.
        :param new_share_network_id: Specify the new share network id.
        """
    result = self._action('migration_check', share_server, {'host': host, 'preserve_snapshots': preserve_snapshots, 'writable': writable, 'nondisruptive': nondisruptive, 'new_share_network_id': new_share_network_id})
    return result[1]