from urllib.parse import quote, urljoin
from django import template
from django.apps import apps
from django.utils.encoding import iri_to_uri
from django.utils.html import conditional_escape
@classmethod
def handle_simple(cls, path):
    if apps.is_installed('django.contrib.staticfiles'):
        from django.contrib.staticfiles.storage import staticfiles_storage
        return staticfiles_storage.url(path)
    else:
        return urljoin(PrefixNode.handle_simple('STATIC_URL'), quote(path))