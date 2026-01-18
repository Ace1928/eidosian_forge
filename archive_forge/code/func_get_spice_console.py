import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
def get_spice_console(self, server, console_type):
    """
        Get a spice console for an instance

        :param server: The :class:`Server` (or its ID) to get console for.
        :param console_type: Type of spice console to get ('spice-html5')
        :returns: An instance of novaclient.base.DictWithMeta
        """
    return self.get_console_url(server, console_type)