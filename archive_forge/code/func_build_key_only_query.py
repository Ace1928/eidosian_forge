import abc
import copy
import functools
import urllib
import warnings
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from oslo_utils import strutils
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.i18n import _
def build_key_only_query(self, params_list):
    """Build a query that does not include values, just keys.

        The Identity API has some calls that define queries without values,
        this can not be accomplished by using urllib.parse.urlencode(). This
        method builds a query using only the keys.
        """
    return '?%s' % '&'.join(params_list) if params_list else ''