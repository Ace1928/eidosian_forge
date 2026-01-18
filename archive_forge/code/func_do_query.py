from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
def do_query(self, table, where, using):
    self.alias_map = {table: self.alias_map[table]}
    self.where = where
    cursor = self.get_compiler(using).execute_sql(CURSOR)
    if cursor:
        with cursor:
            return cursor.rowcount
    return 0