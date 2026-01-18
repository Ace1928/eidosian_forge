from django.db.migrations.utils import field_references
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property
from .base import Operation
@cached_property
def name_lower(self):
    return self.name.lower()