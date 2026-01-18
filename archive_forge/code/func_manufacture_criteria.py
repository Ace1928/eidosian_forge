import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def manufacture_criteria(mapped, values):
    """Given a mapper/class and a namespace of values, produce a WHERE clause.

    The class should be a mapped class and the entries in the dictionary
    correspond to mapped attribute names on the class.

    A value may also be a tuple in which case that particular attribute
    will be compared to a tuple using IN.   The scalar value or
    tuple can also contain None which translates to an IS NULL, that is
    properly joined with OR against an IN expression if appropriate.

    :param cls: a mapped class, or actual :class:`.Mapper` object.

    :param values: dictionary of values.

    """
    mapper = inspect(mapped)
    value_keys = set(values)
    keys = [k for k in mapper.column_attrs.keys() if k in value_keys]
    return sql.and_(*[_sql_crit(mapper.column_attrs[key].expression, values[key]) for key in keys])