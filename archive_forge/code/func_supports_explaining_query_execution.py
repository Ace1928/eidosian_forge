from django.db import ProgrammingError
from django.utils.functional import cached_property
@cached_property
def supports_explaining_query_execution(self):
    """Does this backend support explaining query execution?"""
    return self.connection.ops.explain_prefix is not None