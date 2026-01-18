import copy
import datetime
import warnings
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter
from itertools import chain
from django.forms.utils import to_current_timezone
from django.templatetags.static import static
from django.utils import formats
from django.utils.choices import normalize_choices
from django.utils.dates import MONTHS
from django.utils.formats import get_format
from django.utils.html import format_html, html_safe
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
from .renderers import get_default_renderer
def subwidgets(self, name, value, attrs=None):
    """
        Yield all "subwidgets" of this widget. Used to enable iterating
        options from a BoundField for choice widgets.
        """
    value = self.format_value(value)
    yield from self.options(name, value, attrs)