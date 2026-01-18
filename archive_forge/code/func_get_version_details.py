from urllib import parse as urlparse
from saharaclient.api import base
def get_version_details(self, plugin_name, plugin_version):
    """Get version details

        Get the list of Services and Service Parameters for a specified
        Plugin and Plugin Version.
        """
    return self._get('/plugins/%s/%s' % (plugin_name, plugin_version), 'plugin')