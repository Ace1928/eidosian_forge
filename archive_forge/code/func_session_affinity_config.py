from pprint import pformat
from six import iteritems
import re
@session_affinity_config.setter
def session_affinity_config(self, session_affinity_config):
    """
        Sets the session_affinity_config of this V1ServiceSpec.
        sessionAffinityConfig contains the configurations of session affinity.

        :param session_affinity_config: The session_affinity_config of this
        V1ServiceSpec.
        :type: V1SessionAffinityConfig
        """
    self._session_affinity_config = session_affinity_config