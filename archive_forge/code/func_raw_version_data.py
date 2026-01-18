import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def raw_version_data(self, allow_experimental=False, allow_deprecated=True, allow_unknown=False):
    """Get raw version information from URL.

        Raw data indicates that only minimal validation processing is performed
        on the data, so what is returned here will be the data in the same
        format it was received from the endpoint.

        :param bool allow_experimental: Allow experimental version endpoints.
        :param bool allow_deprecated: Allow deprecated version endpoints.
        :param bool allow_unknown: Allow endpoints with an unrecognised status.

        :returns: The endpoints returned from the server that match the
                  criteria.
        :rtype: list
        """
    versions = []
    for v in self._data:
        try:
            status = v['status']
        except KeyError:
            _LOGGER.warning('Skipping over invalid version data. No stability status in version.')
            continue
        status = status.lower()
        if status in self.CURRENT_STATUSES:
            versions.append(v)
        elif status in self.DEPRECATED_STATUSES:
            if allow_deprecated:
                versions.append(v)
        elif status in self.EXPERIMENTAL_STATUSES:
            if allow_experimental:
                versions.append(v)
        elif allow_unknown:
            versions.append(v)
    return versions