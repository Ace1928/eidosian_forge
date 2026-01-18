from oslo_db import exception as db_exc
from oslo_db.sqlalchemy.utils import paginate_query
from oslo_log import log as logging
import sqlalchemy.exc as sa_exc
from sqlalchemy import or_
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
import glance.db.sqlalchemy.metadef_api as metadef_api
from glance.db.sqlalchemy import models_metadef as models
from glance.i18n import _
def delete_cascade(context, session, name):
    """Raise if not found, has references or not visible"""
    namespace_rec = _get_by_name(context, session, name)
    try:
        metadef_api.tag.delete_namespace_content(context, session, namespace_rec.id)
        metadef_api.object.delete_namespace_content(context, session, namespace_rec.id)
        metadef_api.property.delete_namespace_content(context, session, namespace_rec.id)
        metadef_api.resource_type_association.delete_namespace_content(context, session, namespace_rec.id)
        session.delete(namespace_rec)
        session.flush()
    except db_exc.DBError as e:
        if isinstance(e.inner_exception, sa_exc.IntegrityError):
            LOG.debug('Metadata definition namespace=%s not deleted. Other records still refer to it.', name)
            raise exc.MetadefIntegrityError(record_type='namespace', record_name=name)
        else:
            raise
    return namespace_rec.to_dict()