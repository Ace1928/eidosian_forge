from datetime import datetime, date
from decimal import Decimal
from json import JSONEncoder
def is_saobject(obj):
    return hasattr(obj, '_sa_class_manager')