from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import importutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
def cache_image_iter(self, image_id, image_iter, image_checksum=None):
    """
        Cache an image with supplied iterator.

        :param image_id: Image ID
        :param image_file: Iterator retrieving image chunks
        :param image_checksum: Checksum of image

        :returns: True if image file was cached, False otherwise
        """
    if not self.driver.is_cacheable(image_id):
        return False
    for chunk in self.get_caching_iter(image_id, image_checksum, image_iter):
        pass
    return True