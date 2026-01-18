from pathlib import Path
from django.dispatch import receiver
from django.template import engines
from django.template.backends.django import DjangoTemplates
from django.utils._os import to_path
from django.utils.autoreload import autoreload_started, file_changed, is_django_path
def reset_loaders():
    from django.forms.renderers import get_default_renderer
    for backend in engines.all():
        if not isinstance(backend, DjangoTemplates):
            continue
        for loader in backend.engine.template_loaders:
            loader.reset()
    backend = getattr(get_default_renderer(), 'engine', None)
    if isinstance(backend, DjangoTemplates):
        for loader in backend.engine.template_loaders:
            loader.reset()