import collections.abc
import copy
import errno
import functools
import http.client
import os
import re
import urllib.parse as urlparse
import osprofiler.web
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import netutils
from glance.common import auth
from glance.common import exception
from glance.common import utils
from glance.i18n import _
def handle_redirects(func):
    """
    Wrap the _do_request function to handle HTTP redirects.
    """
    MAX_REDIRECTS = 5

    @functools.wraps(func)
    def wrapped(self, method, url, body, headers):
        for i in range(MAX_REDIRECTS):
            try:
                return func(self, method, url, body, headers)
            except exception.RedirectException as redirect:
                if redirect.url is None:
                    raise exception.InvalidRedirect()
                url = redirect.url
        raise exception.MaxRedirectsExceeded(redirects=MAX_REDIRECTS)
    return wrapped