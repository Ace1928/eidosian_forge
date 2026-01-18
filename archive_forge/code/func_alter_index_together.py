import logging
import operator
from datetime import datetime
from django.conf import settings
from django.db.backends.ddl_references import (
from django.db.backends.utils import names_digest, split_identifier, truncate_name
from django.db.models import NOT_PROVIDED, Deferrable, Index
from django.db.models.sql import Query
from django.db.transaction import TransactionManagementError, atomic
from django.utils import timezone
def alter_index_together(self, model, old_index_together, new_index_together):
    """
        Deal with a model changing its index_together. The input
        index_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
    olds = {tuple(fields) for fields in old_index_together}
    news = {tuple(fields) for fields in new_index_together}
    for fields in olds.difference(news):
        self._delete_composed_index(model, fields, {'index': True, 'unique': False}, self.sql_delete_index)
    for field_names in news.difference(olds):
        fields = [model._meta.get_field(field) for field in field_names]
        self.execute(self._create_index_sql(model, fields=fields, suffix='_idx'))