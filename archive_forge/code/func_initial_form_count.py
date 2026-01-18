from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.forms import ModelForm, modelformset_factory
from django.forms.models import BaseModelFormSet
def initial_form_count(self):
    if self.save_as_new:
        return 0
    return super().initial_form_count()