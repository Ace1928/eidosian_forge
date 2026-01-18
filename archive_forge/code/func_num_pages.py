import collections.abc
import inspect
import warnings
from math import ceil
from django.utils.functional import cached_property
from django.utils.inspect import method_has_no_args
from django.utils.translation import gettext_lazy as _
@cached_property
def num_pages(self):
    """Return the total number of pages."""
    if self.count == 0 and (not self.allow_empty_first_page):
        return 0
    hits = max(1, self.count - self.orphans)
    return ceil(hits / self.per_page)