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
def make_model_name(self, table, snake_case=True):
    if snake_case:
        table = make_snake_case(table)
    model = re.sub('[^\\w]+', '', table)
    model_name = ''.join((sub.title() for sub in model.split('_')))
    if not model_name[0].isalpha():
        model_name = 'T' + model_name
    return model_name