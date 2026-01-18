from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
@property
def app_labeled_name(self):
    model = self.model_class()
    if not model:
        return self.model
    return '%s | %s' % (model._meta.app_config.verbose_name, model._meta.verbose_name)