from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
def get_related_updates(self):
    """
        Return a list of query objects: one for each update required to an
        ancestor model. Each query will have the same filtering conditions as
        the current query but will only update a single table.
        """
    if not self.related_updates:
        return []
    result = []
    for model, values in self.related_updates.items():
        query = UpdateQuery(model)
        query.values = values
        if self.related_ids is not None:
            query.add_filter('pk__in', self.related_ids[model])
        result.append(query)
    return result