from django.core.exceptions import ImproperlyConfigured
from django.forms import Form
from django.forms import models as model_forms
from django.http import HttpResponseRedirect
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
from django.views.generic.detail import (
def form_invalid(self, form):
    """If the form is invalid, render the invalid form."""
    return self.render_to_response(self.get_context_data(form=form))