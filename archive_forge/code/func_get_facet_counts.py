import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
def get_facet_counts(self, pk_attname, filtered_qs):
    lookup_condition = self.get_lookup_condition()
    return {'empty__c': models.Count(pk_attname, filter=lookup_condition), 'not_empty__c': models.Count(pk_attname, filter=~lookup_condition)}