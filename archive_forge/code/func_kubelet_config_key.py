from pprint import pformat
from six import iteritems
import re
@kubelet_config_key.setter
def kubelet_config_key(self, kubelet_config_key):
    """
        Sets the kubelet_config_key of this V1ConfigMapNodeConfigSource.
        KubeletConfigKey declares which key of the referenced ConfigMap
        corresponds to the KubeletConfiguration structure This field is required
        in all cases.

        :param kubelet_config_key: The kubelet_config_key of this
        V1ConfigMapNodeConfigSource.
        :type: str
        """
    if kubelet_config_key is None:
        raise ValueError('Invalid value for `kubelet_config_key`, must not be `None`')
    self._kubelet_config_key = kubelet_config_key