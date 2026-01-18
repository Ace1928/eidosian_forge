from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
def get_for_model(self, model, for_concrete_model=True):
    """
        Return the ContentType object for a given model, creating the
        ContentType if necessary. Lookups are cached so that subsequent lookups
        for the same model don't hit the database.
        """
    opts = self._get_opts(model, for_concrete_model)
    try:
        return self._get_from_cache(opts)
    except KeyError:
        pass
    try:
        ct = self.get(app_label=opts.app_label, model=opts.model_name)
    except self.model.DoesNotExist:
        ct, created = self.get_or_create(app_label=opts.app_label, model=opts.model_name)
    self._add_to_cache(self.db, ct)
    return ct