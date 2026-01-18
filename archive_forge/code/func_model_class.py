from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
def model_class(self):
    """Return the model class for this type of content."""
    try:
        return apps.get_model(self.app_label, self.model)
    except LookupError:
        return None