import abc
import collections
import datetime
from dateutil import tz
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql import yaql_interface
def is_specialization_of(self, other):
    if not isinstance(other, PythonType):
        return False
    try:
        len(self.python_type)
        len(other.python_type)
    except Exception:
        return issubclass(self.python_type, other.python_type) and (not issubclass(other.python_type, self.python_type))
    else:
        return False