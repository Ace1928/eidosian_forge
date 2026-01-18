import collections.abc
import inspect
import warnings
from math import ceil
from django.utils.functional import cached_property
from django.utils.inspect import method_has_no_args
from django.utils.translation import gettext_lazy as _
@property
def page_range(self):
    """
        Return a 1-based range of pages for iterating through within
        a template for loop.
        """
    return range(1, self.num_pages + 1)