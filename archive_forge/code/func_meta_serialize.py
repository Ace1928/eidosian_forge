import collections
import email
from email.mime import multipart
from email.mime import text
import os
import pkgutil
import string
from urllib import parse as urlparse
from neutronclient.common import exceptions as q_exceptions
from novaclient import api_versions
from novaclient import client as nc
from novaclient import exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import microversion_mixin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def meta_serialize(self, metadata):
    """Serialize non-string metadata values before sending them to Nova."""
    if not isinstance(metadata, collections.abc.Mapping):
        raise exception.StackValidationFailed(message=_('nova server metadata needs to be a Map.'))
    return dict(((key, value if isinstance(value, str) else jsonutils.dumps(value)) for key, value in metadata.items()))