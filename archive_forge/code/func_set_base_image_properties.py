import urllib
from oslo_log import log as logging
from oslo_utils import timeutils
from glance.common import exception
from glance.i18n import _, _LE
def set_base_image_properties(properties=None):
    """Sets optional base properties for creating Image.

    :param properties: Input dict to set some base properties
    """
    if isinstance(properties, dict) and len(properties) == 0:
        properties['disk_format'] = 'qcow2'
        properties['container_format'] = 'bare'