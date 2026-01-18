import collections.abc
import inspect
import warnings
from math import ceil
from django.utils.functional import cached_property
from django.utils.inspect import method_has_no_args
from django.utils.translation import gettext_lazy as _
def next_page_number(self):
    return self.paginator.validate_number(self.number + 1)