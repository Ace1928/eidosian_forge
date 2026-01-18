from django.apps import apps as django_apps
from django.conf import settings
from django.core import paginator
from django.core.exceptions import ImproperlyConfigured
from django.utils import translation
def get_languages_for_item(self, item):
    """Languages for which this item is displayed."""
    return self._languages()