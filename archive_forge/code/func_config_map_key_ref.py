from pprint import pformat
from six import iteritems
import re
@config_map_key_ref.setter
def config_map_key_ref(self, config_map_key_ref):
    """
        Sets the config_map_key_ref of this V1EnvVarSource.
        Selects a key of a ConfigMap.

        :param config_map_key_ref: The config_map_key_ref of this
        V1EnvVarSource.
        :type: V1ConfigMapKeySelector
        """
    self._config_map_key_ref = config_map_key_ref