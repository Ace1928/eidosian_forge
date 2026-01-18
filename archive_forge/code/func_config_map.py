from pprint import pformat
from six import iteritems
import re
@config_map.setter
def config_map(self, config_map):
    """
        Sets the config_map of this V1Volume.
        ConfigMap represents a configMap that should populate this volume

        :param config_map: The config_map of this V1Volume.
        :type: V1ConfigMapVolumeSource
        """
    self._config_map = config_map