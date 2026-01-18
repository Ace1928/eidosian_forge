from functools import update_wrapper
from weakref import WeakSet
from django.apps import apps
from django.conf import settings
from django.contrib.admin import ModelAdmin, actions
from django.contrib.admin.exceptions import AlreadyRegistered, NotRegistered
from django.contrib.admin.views.autocomplete import AutocompleteJsonView
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.core.exceptions import ImproperlyConfigured
from django.db.models.base import ModelBase
from django.http import Http404, HttpResponsePermanentRedirect, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import NoReverseMatch, Resolver404, resolve, reverse
from django.utils.decorators import method_decorator
from django.utils.functional import LazyObject
from django.utils.module_loading import import_string
from django.utils.text import capfirst
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from django.views.decorators.cache import never_cache
from django.views.decorators.common import no_append_slash
from django.views.decorators.csrf import csrf_protect
from django.views.i18n import JavaScriptCatalog
@no_append_slash
def catch_all_view(self, request, url):
    if settings.APPEND_SLASH and (not url.endswith('/')):
        urlconf = getattr(request, 'urlconf', None)
        try:
            match = resolve('%s/' % request.path_info, urlconf)
        except Resolver404:
            pass
        else:
            if getattr(match.func, 'should_append_slash', True):
                return HttpResponsePermanentRedirect(request.get_full_path(force_append_slash=True))
    raise Http404