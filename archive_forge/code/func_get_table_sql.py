from collections import namedtuple
from inspect import isclass
import re
import warnings
from peewee import *
from peewee import _StringField
from peewee import _query_val_transform
from peewee import CommaNodeList
from peewee import SCOPE_VALUES
from peewee import make_snake_case
from peewee import text_type
def get_table_sql(model):
    sql, params = model._schema._create_table().query()
    if model._meta.database.param != '%s':
        sql = sql.replace(model._meta.database.param, '%s')
    match_obj = re.match('^(.+?\\()(.+)(\\).*)', sql)
    create, columns, extra = match_obj.groups()
    indented = ',\n'.join(('  %s' % column for column in columns.split(', ')))
    clean = '\n'.join((create, indented, extra)).strip()
    return clean % tuple(map(_query_val_transform, params))