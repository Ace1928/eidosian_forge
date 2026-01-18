import collections
from collections import abc
import itertools
import logging
import re
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import Boolean
from sqlalchemy.engine import Connectable
from sqlalchemy.engine import url as sa_url
from sqlalchemy import exc
from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql.expression import literal_column
from sqlalchemy.sql import text
from sqlalchemy import Table
from oslo_db._i18n import _
from oslo_db import exception
from oslo_db.sqlalchemy import models
def drop_old_duplicate_entries_from_table(engine, table_name, use_soft_delete, *uc_column_names):
    """Drop all old rows having the same values for columns in uc_columns.

    This method drop (or mark ad `deleted` if use_soft_delete is True) old
    duplicate rows form table with name `table_name`.

    :param engine:          Sqlalchemy engine
    :param table_name:      Table with duplicates
    :param use_soft_delete: If True - values will be marked as `deleted`,
                            if False - values will be removed from table
    :param uc_column_names: Unique constraint columns
    """
    meta = MetaData()
    table = Table(table_name, meta, autoload_with=engine)
    columns_for_group_by = [table.c[name] for name in uc_column_names]
    columns_for_select = [func.max(table.c.id)]
    columns_for_select.extend(columns_for_group_by)
    duplicated_rows_select = sqlalchemy.sql.select(*columns_for_select).group_by(*columns_for_group_by).having(func.count(table.c.id) > 1)
    with engine.connect() as conn, conn.begin():
        for row in conn.execute(duplicated_rows_select).fetchall():
            delete_condition = table.c.id != row[0]
            is_none = None
            delete_condition &= table.c.deleted_at == is_none
            for name in uc_column_names:
                delete_condition &= table.c[name] == row._mapping[name]
            rows_to_delete_select = sqlalchemy.sql.select(table.c.id).where(delete_condition)
            for row in conn.execute(rows_to_delete_select).fetchall():
                LOG.info('Deleting duplicated row with id: %(id)s from table: %(table)s', dict(id=row[0], table=table_name))
            if use_soft_delete:
                delete_statement = table.update().where(delete_condition).values({'deleted': literal_column('id'), 'updated_at': literal_column('updated_at'), 'deleted_at': timeutils.utcnow()})
            else:
                delete_statement = table.delete().where(delete_condition)
            conn.execute(delete_statement)