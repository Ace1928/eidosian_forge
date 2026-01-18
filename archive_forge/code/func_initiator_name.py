from pprint import pformat
from six import iteritems
import re
@initiator_name.setter
def initiator_name(self, initiator_name):
    """
        Sets the initiator_name of this V1ISCSIVolumeSource.
        Custom iSCSI Initiator Name. If initiatorName is specified with
        iscsiInterface simultaneously, new iSCSI interface <target
        portal>:<volume name> will be created for the connection.

        :param initiator_name: The initiator_name of this V1ISCSIVolumeSource.
        :type: str
        """
    self._initiator_name = initiator_name