from pprint import pformat
from six import iteritems
import re
@lun.setter
def lun(self, lun):
    """
        Sets the lun of this V1ISCSIVolumeSource.
        iSCSI Target Lun number.

        :param lun: The lun of this V1ISCSIVolumeSource.
        :type: int
        """
    if lun is None:
        raise ValueError('Invalid value for `lun`, must not be `None`')
    self._lun = lun