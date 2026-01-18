from pprint import pformat
from six import iteritems
import re
@pd_name.setter
def pd_name(self, pd_name):
    """
        Sets the pd_name of this V1GCEPersistentDiskVolumeSource.
        Unique name of the PD resource in GCE. Used to identify the disk in GCE.
        More info:
        https://kubernetes.io/docs/concepts/storage/volumes#gcepersistentdisk

        :param pd_name: The pd_name of this V1GCEPersistentDiskVolumeSource.
        :type: str
        """
    if pd_name is None:
        raise ValueError('Invalid value for `pd_name`, must not be `None`')
    self._pd_name = pd_name