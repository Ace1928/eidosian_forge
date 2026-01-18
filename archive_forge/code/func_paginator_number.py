import datetime
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
from django.contrib.admin.views.main import (
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.template import Library
from django.template.loader import get_template
from django.templatetags.static import static
from django.urls import NoReverseMatch
from django.utils import formats, timezone
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.utils.text import capfirst
from django.utils.translation import gettext as _
from .base import InclusionAdminNode
@register.simple_tag
def paginator_number(cl, i):
    """
    Generate an individual page index link in a paginated list.
    """
    if i == cl.paginator.ELLIPSIS:
        return format_html('{} ', cl.paginator.ELLIPSIS)
    elif i == cl.page_num:
        return format_html('<span class="this-page">{}</span> ', i)
    else:
        return format_html('<a href="{}"{}>{}</a> ', cl.get_query_string({PAGE_VAR: i}), mark_safe(' class="end"' if i == cl.paginator.num_pages else ''), i)