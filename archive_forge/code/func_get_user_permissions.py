from django.apps import apps
from django.contrib import auth
from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.hashers import make_password
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import PermissionDenied
from django.core.mail import send_mail
from django.db import models
from django.db.models.manager import EmptyManager
from django.utils import timezone
from django.utils.itercompat import is_iterable
from django.utils.translation import gettext_lazy as _
from .validators import UnicodeUsernameValidator
def get_user_permissions(self, obj=None):
    return _user_get_permissions(self, obj, 'user')