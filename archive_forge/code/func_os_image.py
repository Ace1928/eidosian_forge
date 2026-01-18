from pprint import pformat
from six import iteritems
import re
@os_image.setter
def os_image(self, os_image):
    """
        Sets the os_image of this V1NodeSystemInfo.
        OS Image reported by the node from /etc/os-release (e.g. Debian
        GNU/Linux 7 (wheezy)).

        :param os_image: The os_image of this V1NodeSystemInfo.
        :type: str
        """
    if os_image is None:
        raise ValueError('Invalid value for `os_image`, must not be `None`')
    self._os_image = os_image