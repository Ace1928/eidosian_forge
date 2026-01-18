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
def alter_db_table_comment(self, model, old_db_table_comment, new_db_table_comment):
    if self.sql_alter_table_comment and self.connection.features.supports_comments:
        self.execute(self.sql_alter_table_comment % {'table': self.quote_name(model._meta.db_table), 'comment': self.quote_value(new_db_table_comment or '')})