import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
@api_versions.wraps('2.8')
def get_mks_console(self, server):
    """
        Get a mks console for an instance

        :param server: The :class:`Server` (or its ID) to get console for.
        :returns: An instance of novaclient.base.DictWithMeta
        """
    return self.get_console_url(server, 'webmks')