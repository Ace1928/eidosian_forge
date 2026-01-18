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
def pagination(cl):
    """
    Generate the series of links to the pages in a paginated list.
    """
    pagination_required = (not cl.show_all or not cl.can_show_all) and cl.multi_page
    page_range = cl.paginator.get_elided_page_range(cl.page_num) if pagination_required else []
    need_show_all_link = cl.can_show_all and (not cl.show_all) and cl.multi_page
    return {'cl': cl, 'pagination_required': pagination_required, 'show_all_url': need_show_all_link and cl.get_query_string({ALL_VAR: ''}), 'page_range': page_range, 'ALL_VAR': ALL_VAR, '1': 1}