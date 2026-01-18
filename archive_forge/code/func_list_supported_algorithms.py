from libcloud.common.base import BaseDriver, ConnectionKey
from libcloud.common.types import LibcloudError
def list_supported_algorithms(self):
    """
        Return algorithms supported by this driver.

        :rtype: ``list`` of ``str``
        """
    return list(self._ALGORITHM_TO_VALUE_MAP.keys())