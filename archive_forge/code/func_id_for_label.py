import copy
from itertools import chain
from django import forms
from django.contrib.postgres.validators import (
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from ..utils import prefix_validation_error
def id_for_label(self, id_):
    if id_:
        id_ += '_0'
    return id_