import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
def get_vnc_console(self, server, console_type):
    """
        Get a vnc console for an instance

        :param server: The :class:`Server` (or its ID) to get console for.
        :param console_type: Type of vnc console to get ('novnc' or 'xvpvnc')
        :returns: An instance of novaclient.base.DictWithMeta
        """
    return self.get_console_url(server, console_type)