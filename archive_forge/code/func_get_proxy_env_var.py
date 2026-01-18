import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
def get_proxy_env_var(self, name, default_to='all'):
    """Get a proxy env var.

        Note that we indirectly rely on
        urllib.getproxies_environment taking into account the
        uppercased values for proxy variables.
        """
    try:
        return self.proxies[name.lower()]
    except KeyError:
        if default_to is not None:
            try:
                return self.proxies[default_to]
            except KeyError:
                pass
    return None