from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
def get_for_models(self, *models, for_concrete_models=True):
    """
        Given *models, return a dictionary mapping {model: content_type}.
        """
    results = {}
    needed_models = defaultdict(set)
    needed_opts = defaultdict(list)
    for model in models:
        opts = self._get_opts(model, for_concrete_models)
        try:
            ct = self._get_from_cache(opts)
        except KeyError:
            needed_models[opts.app_label].add(opts.model_name)
            needed_opts[opts].append(model)
        else:
            results[model] = ct
    if needed_opts:
        condition = Q(*(Q(('app_label', app_label), ('model__in', models)) for app_label, models in needed_models.items()), _connector=Q.OR)
        cts = self.filter(condition)
        for ct in cts:
            opts_models = needed_opts.pop(ct._meta.apps.get_model(ct.app_label, ct.model)._meta, [])
            for model in opts_models:
                results[model] = ct
            self._add_to_cache(self.db, ct)
    for opts, opts_models in needed_opts.items():
        ct = self.create(app_label=opts.app_label, model=opts.model_name)
        self._add_to_cache(self.db, ct)
        for model in opts_models:
            results[model] = ct
    return results