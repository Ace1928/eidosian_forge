import concurrent.futures
import urllib.parse
import keystoneauth1.exceptions
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1._proxy import Proxy
def delete_autocreated_image_objects(self, container=None, segment_prefix=None):
    """Delete all objects autocreated for image uploads.

        This method should generally not be needed, as shade should clean up
        the objects it uses for object-based image creation. If something
        goes wrong and it is found that there are leaked objects, this method
        can be used to delete any objects that shade has created on the user's
        behalf in service of image uploads.

        :param str container: Name of the container. Defaults to 'images'.
        :param str segment_prefix: Prefix for the image segment names to
            delete. If not given, all image upload segments present are
            deleted.
        :returns: True if deletion was successful, else False.
        """
    return self.object_store._delete_autocreated_image_objects(container, segment_prefix=segment_prefix)