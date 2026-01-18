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
@register.tag(name='change_list_object_tools')
def change_list_object_tools_tag(parser, token):
    """Display the row of change list object tools."""
    return InclusionAdminNode(parser, token, func=lambda context: context, template_name='change_list_object_tools.html')