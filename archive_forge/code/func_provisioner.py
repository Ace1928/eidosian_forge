from pprint import pformat
from six import iteritems
import re
@provisioner.setter
def provisioner(self, provisioner):
    """
        Sets the provisioner of this V1beta1StorageClass.
        Provisioner indicates the type of the provisioner.

        :param provisioner: The provisioner of this V1beta1StorageClass.
        :type: str
        """
    if provisioner is None:
        raise ValueError('Invalid value for `provisioner`, must not be `None`')
    self._provisioner = provisioner