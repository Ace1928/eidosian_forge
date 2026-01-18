from pprint import pformat
from six import iteritems
import re
@downward_api.setter
def downward_api(self, downward_api):
    """
        Sets the downward_api of this V1Volume.
        DownwardAPI represents downward API about the pod that should populate
        this volume

        :param downward_api: The downward_api of this V1Volume.
        :type: V1DownwardAPIVolumeSource
        """
    self._downward_api = downward_api