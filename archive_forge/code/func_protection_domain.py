from pprint import pformat
from six import iteritems
import re
@protection_domain.setter
def protection_domain(self, protection_domain):
    """
        Sets the protection_domain of this V1ScaleIOVolumeSource.
        The name of the ScaleIO Protection Domain for the configured storage.

        :param protection_domain: The protection_domain of this
        V1ScaleIOVolumeSource.
        :type: str
        """
    self._protection_domain = protection_domain