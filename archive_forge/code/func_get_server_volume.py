import warnings
from novaclient import api_versions
from novaclient import base
def get_server_volume(self, server_id, volume_id=None, attachment_id=None):
    """
        Get the volume identified by the volume ID, that is attached to
        the given server ID

        :param server_id: The ID of the server
        :param volume_id: The ID of the volume to attach
        :rtype: :class:`Volume`
        """
    if attachment_id is not None and volume_id is not None:
        raise TypeError('You cannot specify both volume_id and attachment_id arguments.')
    elif attachment_id is not None:
        warnings.warn('attachment_id argument of volumes.get_server_volume method is deprecated in favor of volume_id.')
        volume_id = attachment_id
    if volume_id is None:
        raise TypeError('volume_id is required argument.')
    return self._get('/servers/%s/os-volume_attachments/%s' % (server_id, volume_id), 'volumeAttachment')