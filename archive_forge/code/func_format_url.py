import base64
import collections.abc
import contextlib
import grp
import hashlib
import itertools
import os
import pwd
import uuid
from cryptography import x509
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import urllib
from keystone.common import password_hashing
import keystone.conf
from keystone import exception
from keystone.i18n import _
def format_url(url, substitutions, silent_keyerror_failures=None):
    """Format a user-defined URL with the given substitutions.

    :param string url: the URL to be formatted
    :param dict substitutions: the dictionary used for substitution
    :param list silent_keyerror_failures: keys for which we should be silent
        if there is a KeyError exception on substitution attempt
    :returns: a formatted URL

    """
    substitutions = WhiteListedItemFilter(WHITELISTED_PROPERTIES, substitutions)
    allow_keyerror = silent_keyerror_failures or []
    try:
        result = url.replace('$(', '%(') % substitutions
    except AttributeError:
        msg = 'Malformed endpoint - %(url)r is not a string'
        LOG.error(msg, {'url': url})
        raise exception.MalformedEndpoint(endpoint=url)
    except KeyError as e:
        if not e.args or e.args[0] not in allow_keyerror:
            msg = 'Malformed endpoint %(url)s - unknown key %(keyerror)s'
            LOG.error(msg, {'url': url, 'keyerror': e})
            raise exception.MalformedEndpoint(endpoint=url)
        else:
            result = None
    except TypeError as e:
        msg = "Malformed endpoint '%(url)s'. The following type error occurred during string substitution: %(typeerror)s"
        LOG.error(msg, {'url': url, 'typeerror': e})
        raise exception.MalformedEndpoint(endpoint=url)
    except ValueError:
        msg = 'Malformed endpoint %s - incomplete format (are you missing a type notifier ?)'
        LOG.error(msg, url)
        raise exception.MalformedEndpoint(endpoint=url)
    return result