from oslo_db import exception as db_exc
from oslo_db.sqlalchemy.utils import paginate_query
from oslo_log import log as logging
from sqlalchemy import func
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
from glance.db.sqlalchemy.metadef_api import namespace as namespace_api
import glance.db.sqlalchemy.metadef_api.utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
from glance.i18n import _LW
def delete_by_namespace_name(context, session, namespace_name):
    namespace = namespace_api.get(context, session, namespace_name)
    return delete_namespace_content(context, session, namespace['id'])