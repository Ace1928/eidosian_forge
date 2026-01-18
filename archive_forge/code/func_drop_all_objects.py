import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
def drop_all_objects(self, engine):
    """Drop all database objects.

        Drops all database objects remaining on the default schema of the
        given engine.

        Per-db implementations will also need to drop items specific to those
        systems, such as sequences, custom types (e.g. pg ENUM), etc.

        """
    with engine.begin() as conn:
        inspector = sqlalchemy.inspect(engine)
        metadata = schema.MetaData()
        tbs = []
        all_fks = []
        for table_name in inspector.get_table_names():
            fks = []
            for fk in inspector.get_foreign_keys(table_name):
                if not fk['name']:
                    continue
                fks.append(schema.ForeignKeyConstraint((), (), name=fk['name']))
            table = schema.Table(table_name, metadata, *fks)
            tbs.append(table)
            all_fks.extend(fks)
        if self.supports_drop_fk:
            for fkc in all_fks:
                conn.execute(schema.DropConstraint(fkc))
        for table in tbs:
            conn.execute(schema.DropTable(table))
        self.drop_additional_objects(conn)