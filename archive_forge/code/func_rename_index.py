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
def rename_index(self, model, old_index, new_index):
    if self.connection.features.can_rename_index:
        self.execute(self._rename_index_sql(model, old_index.name, new_index.name), params=None)
    else:
        self.remove_index(model, old_index)
        self.add_index(model, new_index)