from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
def insert_values(self, fields, objs, raw=False):
    self.fields = fields
    self.objs = objs
    self.raw = raw