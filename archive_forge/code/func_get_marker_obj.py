import functools
from oslo_db import exception as db_exc
from oslo_utils import excutils
import sqlalchemy
from sqlalchemy.ext import associationproxy
from sqlalchemy.orm import exc
from sqlalchemy.orm import properties
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib import exceptions as n_exc
def get_marker_obj(plugin, context, resource, limit, marker):
    """Retrieve a resource marker object.

    This function is used to invoke
    plugin._get_<resource>(context, marker) and is used for pagination.

    :param plugin: The plugin processing the request.
    :param context: The request context.
    :param resource: The resource name.
    :param limit: Indicates if pagination is in effect.
    :param marker: The id of the marker object.
    :returns: The marker object associated with the plugin if limit and marker
        are given.
    """
    if limit and marker:
        return getattr(plugin, '_get_%s' % resource)(context, marker)