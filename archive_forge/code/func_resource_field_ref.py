from pprint import pformat
from six import iteritems
import re
@resource_field_ref.setter
def resource_field_ref(self, resource_field_ref):
    """
        Sets the resource_field_ref of this V1DownwardAPIVolumeFile.
        Selects a resource of the container: only resources limits and requests
        (limits.cpu, limits.memory, requests.cpu and requests.memory) are
        currently supported.

        :param resource_field_ref: The resource_field_ref of this
        V1DownwardAPIVolumeFile.
        :type: V1ResourceFieldSelector
        """
    self._resource_field_ref = resource_field_ref