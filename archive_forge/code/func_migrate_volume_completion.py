from cinderclient.apiclient import base as common_base
from cinderclient import base
def migrate_volume_completion(self, old_volume, new_volume, error):
    """Complete the migration from the old volume to the temp new one.

        :param old_volume: The original :class:`Volume` in the migration
        :param new_volume: The new temporary :class:`Volume` in the migration
        :param error: Inform of an error to cause migration cleanup
        """
    new_volume_id = base.getid(new_volume)
    resp, body = self._action('os-migrate_volume_completion', old_volume, {'new_volume': new_volume_id, 'error': error})
    return common_base.DictWithMeta(body, resp)