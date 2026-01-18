import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def manufacture_entity_criteria(entity, include_only=None, exclude=None):
    """Given a mapped instance, produce a WHERE clause.

    The attributes set upon the instance will be combined to produce
    a SQL expression using the mapped SQL expressions as the base
    of comparison.

    Values on the instance may be set as tuples in which case the
    criteria will produce an IN clause.  None is also acceptable as a
    scalar or tuple entry, which will produce IS NULL that is properly
    joined with an OR against an IN expression if appropriate.

    :param entity: a mapped entity.

    :param include_only: optional sequence of keys to limit which
     keys are included.

    :param exclude: sequence of keys to exclude

    """
    state = inspect(entity)
    exclude = set(exclude) if exclude is not None else set()
    existing = dict(((attr.key, attr.loaded_value) for attr in state.attrs if attr.loaded_value is not orm.attributes.NO_VALUE and attr.key not in exclude))
    if include_only:
        existing = dict(((k, existing[k]) for k in set(existing).intersection(include_only)))
    return manufacture_criteria(state.mapper, existing)