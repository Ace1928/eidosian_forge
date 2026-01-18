import inspect
import re
from asgiref.sync import sync_to_async
from django.apps import apps as django_apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.middleware.csrf import rotate_token
from django.utils.crypto import constant_time_compare
from django.utils.module_loading import import_string
from django.views.decorators.debug import sensitive_variables
from .signals import user_logged_in, user_logged_out, user_login_failed
def get_user_model():
    """
    Return the User model that is active in this project.
    """
    try:
        return django_apps.get_model(settings.AUTH_USER_MODEL, require_ready=False)
    except ValueError:
        raise ImproperlyConfigured("AUTH_USER_MODEL must be of the form 'app_label.model_name'")
    except LookupError:
        raise ImproperlyConfigured("AUTH_USER_MODEL refers to model '%s' that has not been installed" % settings.AUTH_USER_MODEL)