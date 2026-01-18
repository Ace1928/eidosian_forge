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
def password_change(self, request, extra_context=None):
    """
        Handle the "change password" task -- both form display and validation.
        """
    from django.contrib.admin.forms import AdminPasswordChangeForm
    from django.contrib.auth.views import PasswordChangeView
    url = reverse('admin:password_change_done', current_app=self.name)
    defaults = {'form_class': AdminPasswordChangeForm, 'success_url': url, 'extra_context': {**self.each_context(request), **(extra_context or {})}}
    if self.password_change_template is not None:
        defaults['template_name'] = self.password_change_template
    request.current_app = self.name
    return PasswordChangeView.as_view(**defaults)(request)