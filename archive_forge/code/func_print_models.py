import datetime
import os
import sys
from getpass import getpass
from optparse import OptionParser
from peewee import *
from peewee import print_
from peewee import __version__ as peewee_version
from playhouse.cockroachdb import CockroachDatabase
from playhouse.reflection import *
def print_models(introspector, tables=None, preserve_order=False, include_views=False, ignore_unknown=False, snake_case=True):
    database = introspector.introspect(table_names=tables, include_views=include_views, snake_case=snake_case)
    db_kwargs = introspector.get_database_kwargs()
    header = HEADER % (introspector.get_additional_imports(), introspector.get_database_class().__name__, introspector.get_database_name(), ', **%s' % repr(db_kwargs) if db_kwargs else '')
    print_(header)
    if not ignore_unknown:
        print_(UNKNOWN_FIELD)
    print_(BASE_MODEL)

    def _print_table(table, seen, accum=None):
        accum = accum or []
        foreign_keys = database.foreign_keys[table]
        for foreign_key in foreign_keys:
            dest = foreign_key.dest_table
            if dest in accum and table not in accum:
                print_('# Possible reference cycle: %s' % dest)
            if dest not in seen and dest not in accum:
                seen.add(dest)
                if dest != table:
                    _print_table(dest, seen, accum + [table])
        print_('class %s(BaseModel):' % database.model_names[table])
        columns = database.columns[table].items()
        if not preserve_order:
            columns = sorted(columns)
        primary_keys = database.primary_keys[table]
        for name, column in columns:
            skip = all([name in primary_keys, name == 'id', len(primary_keys) == 1, column.field_class in introspector.pk_classes])
            if skip:
                continue
            if column.primary_key and len(primary_keys) > 1:
                column.primary_key = False
            is_unknown = column.field_class is UnknownField
            if is_unknown and ignore_unknown:
                disp = '%s - %s' % (column.name, column.raw_column_type or '?')
                print_('    # %s' % disp)
            else:
                print_('    %s' % column.get_field())
        print_('')
        print_('    class Meta:')
        print_("        table_name = '%s'" % table)
        multi_column_indexes = database.multi_column_indexes(table)
        if multi_column_indexes:
            print_('        indexes = (')
            for fields, unique in sorted(multi_column_indexes):
                print_('            ((%s), %s),' % (', '.join(("'%s'" % field for field in fields)), unique))
            print_('        )')
        if introspector.schema:
            print_("        schema = '%s'" % introspector.schema)
        if len(primary_keys) > 1:
            pk_field_names = sorted([field.name for col, field in columns if col in primary_keys])
            pk_list = ', '.join(("'%s'" % pk for pk in pk_field_names))
            print_('        primary_key = CompositeKey(%s)' % pk_list)
        elif not primary_keys:
            print_('        primary_key = False')
        print_('')
        seen.add(table)
    seen = set()
    for table in sorted(database.model_names.keys()):
        if table not in seen:
            if not tables or table in tables:
                _print_table(table, seen)