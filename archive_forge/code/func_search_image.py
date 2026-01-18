from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
def search_image(self, image, **kwargs):
    """Retrieves list of images based on image name and image_driver name

        :returns: A list of images based on the search query
         i.e., image_name & image_driver

        """
    image_query = {}
    for key, value in kwargs.items():
        if key in IMAGE_SEARCH_ATTRIBUTES:
            image_query[key] = value
        else:
            raise exceptions.InvalidAttribute('Key must be in %s' % ','.join(IMAGE_SEARCH_ATTRIBUTES))
    return self._search(self._path(image) + '/search', image_query)