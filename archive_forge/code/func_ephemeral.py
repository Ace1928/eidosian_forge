from oslo_utils import strutils
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import utils
@property
def ephemeral(self):
    """Provide a user-friendly accessor to OS-FLV-EXT-DATA:ephemeral."""
    return self._info.get('OS-FLV-EXT-DATA:ephemeral', 'N/A')