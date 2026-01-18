import datetime
import decimal
import json
from collections import defaultdict
from functools import reduce
from operator import or_
from django.core.exceptions import FieldDoesNotExist
from django.db import models, router
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import Collector
from django.forms.utils import pretty_name
from django.urls import NoReverseMatch, reverse
from django.utils import formats, timezone
from django.utils.hashable import make_hashable
from django.utils.html import format_html
from django.utils.regex_helper import _lazy_re_compile
from django.utils.text import capfirst
from django.utils.translation import ngettext
from django.utils.translation import override as translation_override
def label_for_field(name, model, model_admin=None, return_attr=False, form=None):
    """
    Return a sensible label for a field name. The name can be a callable,
    property (but not created with @property decorator), or the name of an
    object's attribute, as well as a model field. If return_attr is True, also
    return the resolved attribute (which could be a callable). This will be
    None if (and only if) the name refers to a field.
    """
    attr = None
    try:
        field = _get_non_gfk_field(model._meta, name)
        try:
            label = field.verbose_name
        except AttributeError:
            label = field.related_model._meta.verbose_name
    except FieldDoesNotExist:
        if name == '__str__':
            label = str(model._meta.verbose_name)
            attr = str
        else:
            if callable(name):
                attr = name
            elif hasattr(model_admin, name):
                attr = getattr(model_admin, name)
            elif hasattr(model, name):
                attr = getattr(model, name)
            elif form and name in form.fields:
                attr = form.fields[name]
            else:
                message = "Unable to lookup '%s' on %s" % (name, model._meta.object_name)
                if model_admin:
                    message += ' or %s' % model_admin.__class__.__name__
                if form:
                    message += ' or %s' % form.__class__.__name__
                raise AttributeError(message)
            if hasattr(attr, 'short_description'):
                label = attr.short_description
            elif isinstance(attr, property) and hasattr(attr, 'fget') and hasattr(attr.fget, 'short_description'):
                label = attr.fget.short_description
            elif callable(attr):
                if attr.__name__ == '<lambda>':
                    label = '--'
                else:
                    label = pretty_name(attr.__name__)
            else:
                label = pretty_name(name)
    except FieldIsAForeignKeyColumnName:
        label = pretty_name(name)
        attr = name
    if return_attr:
        return (label, attr)
    else:
        return label