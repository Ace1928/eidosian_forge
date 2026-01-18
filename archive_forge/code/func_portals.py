from pprint import pformat
from six import iteritems
import re
@portals.setter
def portals(self, portals):
    """
        Sets the portals of this V1ISCSIVolumeSource.
        iSCSI Target Portal List. The portal is either an IP or ip_addr:port if
        the port is other than default (typically TCP ports 860 and 3260).

        :param portals: The portals of this V1ISCSIVolumeSource.
        :type: list[str]
        """
    self._portals = portals