import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
@utils.check_resource('image')
def remove_image(self, image, force=False, noprune=False):
    """
        Remove an image. Similar to the ``docker rmi`` command.

        Args:
            image (str): The image to remove
            force (bool): Force removal of the image
            noprune (bool): Do not delete untagged parents
        """
    params = {'force': force, 'noprune': noprune}
    res = self._delete(self._url('/images/{0}', image), params=params)
    return self._result(res, True)