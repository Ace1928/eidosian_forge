import inspect
from importlib import import_module
from inspect import cleandoc
from pathlib import Path
from django.apps import apps
from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.admindocs import utils
from django.contrib.admindocs.utils import (
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.db import models
from django.http import Http404
from django.template.engine import Engine
from django.urls import get_mod_func, get_resolver, get_urlconf
from django.utils._os import safe_join
from django.utils.decorators import method_decorator
from django.utils.functional import cached_property
from django.utils.inspect import (
from django.utils.translation import gettext as _
from django.views.generic import TemplateView
from .utils import get_view_name
def get_readable_field_data_type(field):
    """
    Return the description for a given field type, if it exists. Fields'
    descriptions can contain format strings, which will be interpolated with
    the values of field.__dict__ before being output.
    """
    return field.description % field.__dict__