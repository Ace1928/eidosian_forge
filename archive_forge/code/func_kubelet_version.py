from pprint import pformat
from six import iteritems
import re
@kubelet_version.setter
def kubelet_version(self, kubelet_version):
    """
        Sets the kubelet_version of this V1NodeSystemInfo.
        Kubelet Version reported by the node.

        :param kubelet_version: The kubelet_version of this V1NodeSystemInfo.
        :type: str
        """
    if kubelet_version is None:
        raise ValueError('Invalid value for `kubelet_version`, must not be `None`')
    self._kubelet_version = kubelet_version