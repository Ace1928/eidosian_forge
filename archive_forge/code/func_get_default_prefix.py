from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.forms import ModelForm, modelformset_factory
from django.forms.models import BaseModelFormSet
@classmethod
def get_default_prefix(cls):
    opts = cls.model._meta
    return opts.app_label + '-' + opts.model_name + '-' + cls.ct_field.name + '-' + cls.ct_fk_field.name