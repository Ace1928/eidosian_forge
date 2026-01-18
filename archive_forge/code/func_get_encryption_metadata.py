from cinderclient.apiclient import base as common_base
from cinderclient import base
def get_encryption_metadata(self, volume_id):
    """
        Retrieve the encryption metadata from the desired volume.

        :param volume_id: the id of the volume to query
        :return: a dictionary of volume encryption metadata
        """
    metadata = self._get('/volumes/%s/encryption' % volume_id)
    return common_base.DictWithMeta(metadata._info, metadata.request_ids)