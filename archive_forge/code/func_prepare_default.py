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
def prepare_default(self, value):
    """
        Only used for backends which have requires_literal_defaults feature
        """
    raise NotImplementedError('subclasses of BaseDatabaseSchemaEditor for backends which have requires_literal_defaults must provide a prepare_default() method')