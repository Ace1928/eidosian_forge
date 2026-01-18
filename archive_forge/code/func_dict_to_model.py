import threading
from peewee import *
from peewee import Alias
from peewee import CompoundSelectQuery
from peewee import Metadata
from peewee import callable_
from peewee import __deprecated__
def dict_to_model(model_class, data, ignore_unknown=False):
    return update_model_from_dict(model_class(), data, ignore_unknown)