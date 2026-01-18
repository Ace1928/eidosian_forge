from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
def add_update_fields(self, values_seq):
    """
        Append a sequence of (field, model, value) triples to the internal list
        that will be used to generate the UPDATE query. Might be more usefully
        called add_update_targets() to hint at the extra information here.
        """
    for field, model, val in values_seq:
        if field.generated:
            continue
        if hasattr(val, 'resolve_expression'):
            val = val.resolve_expression(self, allow_joins=False, for_save=True)
        self.values.append((field, model, val))