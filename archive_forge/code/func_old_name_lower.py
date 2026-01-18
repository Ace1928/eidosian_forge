from django.db.migrations.utils import field_references
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property
from .base import Operation
@cached_property
def old_name_lower(self):
    return self.old_name.lower()