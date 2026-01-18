from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.forms import ModelForm, modelformset_factory
from django.forms.models import BaseModelFormSet
def save_new(self, form, commit=True):
    setattr(form.instance, self.ct_field.get_attname(), ContentType.objects.get_for_model(self.instance).pk)
    setattr(form.instance, self.ct_fk_field.get_attname(), self.instance.pk)
    return form.save(commit=commit)