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
def query_with_hooks(context, model, field=None, lazy_fields=None):
    """Query with hooks using the said context and model.

    :param context: The context to use for the DB session.
    :param model: The model to query.
    :param field: The column.
    :param lazy_fields: list of fields for lazy loading
    :returns: The query with hooks applied to it.
    """
    group_by = None
    if field:
        if hasattr(model, field):
            field = getattr(model, field)
        else:
            msg = _("'%s' is not supported as field") % field
            raise n_exc.InvalidInput(error_message=msg)
        query = context.session.query(field)
    else:
        query = context.session.query(model)
    query_filter = None
    if db_utils.model_query_scope_is_project(context, model):
        if hasattr(model, 'rbac_entries'):
            query = query.outerjoin(model.rbac_entries)
            rbac_model = model.rbac_entries.property.mapper.class_
            query_filter = (model.tenant_id == context.tenant_id) | rbac_model.action.in_([constants.ACCESS_SHARED, constants.ACCESS_READONLY]) & ((rbac_model.target_project == context.tenant_id) | (rbac_model.target_project == '*'))
            group_by = model.id
        elif hasattr(model, 'shared'):
            query_filter = (model.tenant_id == context.tenant_id) | (model.shared == sql.true())
        else:
            query_filter = model.tenant_id == context.tenant_id
    for hook in get_hooks(model):
        query_hook = helpers.resolve_ref(hook.get('query'))
        if query_hook:
            query = query_hook(context, model, query)
        filter_hook = helpers.resolve_ref(hook.get('filter'))
        if filter_hook:
            query_filter = filter_hook(context, model, query_filter)
    if query_filter is not None:
        query = query.filter(query_filter)
    if group_by:
        query = query.group_by(group_by)
    if lazy_fields:
        for field in lazy_fields:
            query = query.options(lazyload(field))
    return query