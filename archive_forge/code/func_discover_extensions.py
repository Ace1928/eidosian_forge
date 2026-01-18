import logging
import urllib.parse as urlparse
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient.i18n import _
def discover_extensions(self, url=None):
    """Discover Keystone extensions supported.

        :param url: optional url to test (should have a version in it)

        Returns::

            {
                'message': 'Keystone extensions at http://127.0.0.1:35357/v2',
                'OS-KSEC2': 'OpenStack EC2 Credentials Extension',
            }

        """
    if url:
        return self._check_keystone_extensions(url)