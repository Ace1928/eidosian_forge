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
def each_context(self, request):
    """
        Return a dictionary of variables to put in the template context for
        *every* page in the admin site.

        For sites running on a subpath, use the SCRIPT_NAME value if site_url
        hasn't been customized.
        """
    script_name = request.META['SCRIPT_NAME']
    site_url = script_name if self.site_url == '/' and script_name else self.site_url
    return {'site_title': self.site_title, 'site_header': self.site_header, 'site_url': site_url, 'has_permission': self.has_permission(request), 'available_apps': self.get_app_list(request), 'is_popup': False, 'is_nav_sidebar_enabled': self.enable_nav_sidebar, 'log_entries': self.get_log_entries(request)}