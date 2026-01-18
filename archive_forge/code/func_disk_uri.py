from pprint import pformat
from six import iteritems
import re
@disk_uri.setter
def disk_uri(self, disk_uri):
    """
        Sets the disk_uri of this V1AzureDiskVolumeSource.
        The URI the data disk in the blob storage

        :param disk_uri: The disk_uri of this V1AzureDiskVolumeSource.
        :type: str
        """
    if disk_uri is None:
        raise ValueError('Invalid value for `disk_uri`, must not be `None`')
    self._disk_uri = disk_uri