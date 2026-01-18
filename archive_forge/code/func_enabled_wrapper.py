from django import shortcuts
import django.conf
from six import wraps
from six.moves.urllib import parse
from oauth2client.contrib import django_util
@wraps(wrapped_function)
def enabled_wrapper(request, *args, **kwargs):
    return_url = decorator_kwargs.pop('return_url', request.get_full_path())
    user_oauth = django_util.UserOAuth2(request, scopes, return_url)
    setattr(request, django_util.oauth2_settings.request_prefix, user_oauth)
    return wrapped_function(request, *args, **kwargs)