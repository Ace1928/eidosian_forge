import os.path
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import utils
from glance.i18n import _
def get_image_filepath(self, image_id, cache_status='active'):
    """
        This crafts an absolute path to a specific entry

        :param image_id: Image ID
        :param cache_status: Status of the image in the cache
        """
    if cache_status == 'active':
        return os.path.join(self.base_dir, str(image_id))
    return os.path.join(self.base_dir, cache_status, str(image_id))