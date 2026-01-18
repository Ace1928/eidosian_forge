from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient.v3 import volumes_base
@api_versions.wraps('3.68')
def reimage(self, volume, image_id, reimage_reserved=False):
    """Reimage a volume

        .. warning:: This is a destructive action and the contents of the
            volume will be lost.

        :param volume: Volume to reimage.
        :param reimage_reserved: Boolean to enable or disable reimage
            of a volume that is in 'reserved' state otherwise only
            volumes in 'available' status may be re-imaged.
        :param image_id: The image id.
        """
    return self._action('os-reimage', volume, {'image_id': image_id, 'reimage_reserved': reimage_reserved})