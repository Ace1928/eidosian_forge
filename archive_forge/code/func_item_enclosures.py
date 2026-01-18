from inspect import getattr_static, unwrap
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ImproperlyConfigured, ObjectDoesNotExist
from django.http import Http404, HttpResponse
from django.template import TemplateDoesNotExist, loader
from django.utils import feedgenerator
from django.utils.encoding import iri_to_uri
from django.utils.html import escape
from django.utils.http import http_date
from django.utils.timezone import get_default_timezone, is_naive, make_aware
from django.utils.translation import get_language
def item_enclosures(self, item):
    enc_url = self._get_dynamic_attr('item_enclosure_url', item)
    if enc_url:
        enc = feedgenerator.Enclosure(url=str(enc_url), length=str(self._get_dynamic_attr('item_enclosure_length', item)), mime_type=str(self._get_dynamic_attr('item_enclosure_mime_type', item)))
        return [enc]
    return []