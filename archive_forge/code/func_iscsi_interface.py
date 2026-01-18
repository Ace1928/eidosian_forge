from pprint import pformat
from six import iteritems
import re
@iscsi_interface.setter
def iscsi_interface(self, iscsi_interface):
    """
        Sets the iscsi_interface of this V1ISCSIVolumeSource.
        iSCSI Interface Name that uses an iSCSI transport. Defaults to 'default'
        (tcp).

        :param iscsi_interface: The iscsi_interface of this V1ISCSIVolumeSource.
        :type: str
        """
    self._iscsi_interface = iscsi_interface