import datetime
import decimal
from .base import Database
def bind_parameter(self, cursor):
    self.bound_param = cursor.cursor.var(self.db_type)
    return self.bound_param