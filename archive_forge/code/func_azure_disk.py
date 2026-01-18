from pprint import pformat
from six import iteritems
import re
@azure_disk.setter
def azure_disk(self, azure_disk):
    """
        Sets the azure_disk of this V1PersistentVolumeSpec.
        AzureDisk represents an Azure Data Disk mount on the host and bind mount
        to the pod.

        :param azure_disk: The azure_disk of this V1PersistentVolumeSpec.
        :type: V1AzureDiskVolumeSource
        """
    self._azure_disk = azure_disk