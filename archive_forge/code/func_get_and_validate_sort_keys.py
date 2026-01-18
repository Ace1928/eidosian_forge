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
def get_and_validate_sort_keys(sorts, model):
    """Extract sort keys from sorts and ensure they are valid for the model.

    :param sorts: A list of (key, direction) tuples.
    :param model: A sqlalchemy ORM model class.
    :returns: A list of the extracted sort keys.
    :raises BadRequest: If a sort key attribute references another resource
        and cannot be used in the sort.
    """
    sort_keys = [s[0] for s in sorts]
    for sort_key in sort_keys:
        try:
            sort_key_attr = getattr(model, sort_key)
        except AttributeError as e:
            msg = _("'%s' is an invalid attribute for sort key") % sort_key
            raise n_exc.BadRequest(resource=model.__tablename__, msg=msg) from e
        if isinstance(sort_key_attr.property, properties.RelationshipProperty):
            msg = _("Attribute '%(attr)s' references another resource and cannot be used to sort '%(resource)s' resources") % {'attr': sort_key, 'resource': model.__tablename__}
            raise n_exc.BadRequest(resource=model.__tablename__, msg=msg)
    return sort_keys