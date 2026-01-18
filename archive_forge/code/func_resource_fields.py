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
def resource_fields(resource, fields):
    """Return only the resource items that are in fields.

    :param resource: A resource dict.
    :param fields: A list of fields to select from the resource.
    :returns: A new dict that contains only fields from resource as well
        as its attribute project info.
    """
    if fields:
        resource = {key: item for key, item in resource.items() if key in fields}
    return attributes.populate_project_info(resource)