import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def manufacture_persistent_object(session, specimen, values=None, primary_key=None):
    """Make an ORM-mapped object persistent in a Session without SQL.

    The persistent object is returned.

    If a matching object is already present in the given session, the specimen
    is merged into it and the persistent object returned.  Otherwise, the
    specimen itself is made persistent and is returned.

    The object must contain a full primary key, or provide it via the values or
    primary_key parameters.  The object is peristed to the Session in a "clean"
    state with no pending changes.

    :param session: A Session object.

    :param specimen: a mapped object which is typically transient.

    :param values: a dictionary of values to be applied to the specimen,
     in addition to the state that's already on it.  The attributes will be
     set such that no history is created; the object remains clean.

    :param primary_key: optional tuple-based primary key.  This will also
     be applied to the instance if present.


    """
    state = inspect(specimen)
    mapper = state.mapper
    for k, v in values.items():
        orm.attributes.set_committed_value(specimen, k, v)
    pk_attrs = [mapper.get_property_by_column(col).key for col in mapper.primary_key]
    if primary_key is not None:
        for key, value in zip(pk_attrs, primary_key):
            orm.attributes.set_committed_value(specimen, key, value)
    for key in pk_attrs:
        if state.attrs[key].loaded_value is orm.attributes.NO_VALUE:
            raise ValueError('full primary key must be present')
    orm.make_transient_to_detached(specimen)
    if state.key not in session.identity_map:
        session.add(specimen)
        return specimen
    else:
        return session.merge(specimen, load=False)