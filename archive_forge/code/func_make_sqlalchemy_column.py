import datetime
import logging
from petl.compat import long, text_type
from petl.errors import ArgumentError
from petl.util.materialise import columns
from petl.transform.basics import head
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
def make_sqlalchemy_column(col, colname, constraints=True):
    """
    Infer an appropriate SQLAlchemy column type based on a sequence of values.

    Keyword arguments:

    col : sequence
        A sequence of values to use to infer type, length etc.
    colname : string
        Name of column
    constraints : bool
        If True use length and nullable constraints

    """
    import sqlalchemy
    col_not_none = [v for v in col if v is not None]
    sql_column_kwargs = {}
    sql_type_kwargs = {}
    if len(col_not_none) == 0:
        sql_column_type = sqlalchemy.String
        if constraints:
            sql_type_kwargs['length'] = NULL_COLUMN_MAX_LENGTH
    elif all((isinstance(v, bool) for v in col_not_none)):
        sql_column_type = sqlalchemy.Boolean
    elif all((isinstance(v, int) for v in col_not_none)):
        if max(col_not_none) > SQL_INTEGER_MAX or min(col_not_none) < SQL_INTEGER_MIN:
            sql_column_type = sqlalchemy.BigInteger
        else:
            sql_column_type = sqlalchemy.Integer
    elif all((isinstance(v, long) for v in col_not_none)):
        sql_column_type = sqlalchemy.BigInteger
    elif all((isinstance(v, (int, long)) for v in col_not_none)):
        sql_column_type = sqlalchemy.BigInteger
    elif all((isinstance(v, (int, long, float)) for v in col_not_none)):
        sql_column_type = sqlalchemy.Float
    elif all((isinstance(v, datetime.datetime) for v in col_not_none)):
        sql_column_type = sqlalchemy.DateTime
    elif all((isinstance(v, datetime.date) for v in col_not_none)):
        sql_column_type = sqlalchemy.Date
    elif all((isinstance(v, datetime.time) for v in col_not_none)):
        sql_column_type = sqlalchemy.Time
    else:
        sql_column_type = sqlalchemy.String
        if constraints:
            sql_type_kwargs['length'] = max([len(text_type(v)) for v in col])
    if constraints:
        sql_column_kwargs['nullable'] = len(col_not_none) < len(col)
    return sqlalchemy.Column(colname, sql_column_type(**sql_type_kwargs), **sql_column_kwargs)