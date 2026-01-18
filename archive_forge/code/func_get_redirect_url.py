from urllib.parse import urlparse, urlunparse
from django.conf import settings
from django.contrib.auth import REDIRECT_FIELD_NAME, get_user_model
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import (
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.http import HttpResponseRedirect, QueryDict
from django.shortcuts import resolve_url
from django.urls import reverse_lazy
from django.utils.decorators import method_decorator
from django.utils.http import url_has_allowed_host_and_scheme, urlsafe_base64_decode
from django.utils.translation import gettext_lazy as _
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.debug import sensitive_post_parameters
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView
def get_redirect_url(self):
    """Return the user-originating redirect URL if it's safe."""
    redirect_to = self.request.POST.get(self.redirect_field_name, self.request.GET.get(self.redirect_field_name))
    url_is_safe = url_has_allowed_host_and_scheme(url=redirect_to, allowed_hosts=self.get_success_url_allowed_hosts(), require_https=self.request.is_secure())
    return redirect_to if url_is_safe else ''