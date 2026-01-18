import abc
import copy
from http import client as http_client
from urllib import parse as urlparse
from oslo_utils import strutils
from ironicclient.common.apiclient import exceptions
from ironicclient.common.i18n import _
@property
def human_id(self):
    """Human-readable ID which can be used for bash completion."""
    if self.HUMAN_ID:
        name = getattr(self, self.NAME_ATTR, None)
        if name is not None:
            return strutils.to_slug(name)
    return None