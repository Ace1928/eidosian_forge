import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
def import_image_from_image(self, image, repository=None, tag=None, changes=None):
    """
        Like :py:meth:`~docker.api.image.ImageApiMixin.import_image`, but only
        supports importing from another image, like the ``FROM`` Dockerfile
        parameter.

        Args:
            image (str): Image name to import from
            repository (str): The repository to create
            tag (str): The tag to apply
        """
    return self.import_image(image=image, repository=repository, tag=tag, changes=changes)