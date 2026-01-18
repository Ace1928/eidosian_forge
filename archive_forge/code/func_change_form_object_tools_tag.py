import json
from django import template
from django.template.context import Context
from .base import InclusionAdminNode
@register.tag(name='change_form_object_tools')
def change_form_object_tools_tag(parser, token):
    """Display the row of change form object tools."""
    return InclusionAdminNode(parser, token, func=lambda context: context, template_name='change_form_object_tools.html')