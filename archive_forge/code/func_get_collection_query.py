from oslo_db.sqlalchemy import utils as sa_utils
from sqlalchemy.orm import lazyload
from sqlalchemy import sql, or_, and_
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib import constants
from neutron_lib.db import utils as db_utils
from neutron_lib import exceptions as n_exc
from neutron_lib.objects import utils as obj_utils
from neutron_lib.utils import helpers
def get_collection_query(context, model, filters=None, sorts=None, limit=None, marker_obj=None, page_reverse=False, field=None, lazy_fields=None):
    """Get a collection query.

    :param context: The context to use for the DB session.
    :param model: The model to use.
    :param filters: The filters to apply in the query.
    :param sorts: The sort keys to use.
    :param limit: The limit associated with the query.
    :param marker_obj: The marker object if applicable.
    :param page_reverse: If reverse paging should be used.
    :param field: Column, in string format, from the "model"; the query will
                  return only this parameter instead of the full model columns.
    :param lazy_fields: list of fields for lazy loading
    :returns: A paginated query for the said model.
    """
    collection = query_with_hooks(context, model, field=field, lazy_fields=lazy_fields)
    collection = apply_filters(collection, model, filters, context)
    if sorts:
        sort_keys = db_utils.get_and_validate_sort_keys(sorts, model)
        sort_dirs = db_utils.get_sort_dirs(sorts, page_reverse)
        for k in _unique_keys(model):
            if k not in sort_keys:
                sort_keys.append(k)
                sort_dirs.append('asc')
        collection = sa_utils.paginate_query(collection, model, limit, marker=marker_obj, sort_keys=sort_keys, sort_dirs=sort_dirs)
    return collection