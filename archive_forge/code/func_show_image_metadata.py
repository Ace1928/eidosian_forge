from cinderclient.apiclient import base as common_base
from cinderclient import base
def show_image_metadata(self, volume):
    """Show a volume's image metadata.

        :param volume : The :class: `Volume` where the image metadata
            associated.
        """
    return self._action('os-show_image_metadata', volume)