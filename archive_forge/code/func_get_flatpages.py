from django import template
from django.conf import settings
from django.contrib.flatpages.models import FlatPage
from django.contrib.sites.shortcuts import get_current_site
@register.tag
def get_flatpages(parser, token):
    """
    Retrieve all flatpage objects available for the current site and
    visible to the specific user (or visible to all users if no user is
    specified). Populate the template context with them in a variable
    whose name is defined by the ``as`` clause.

    An optional ``for`` clause controls the user whose permissions are used in
    determining which flatpages are visible.

    An optional argument, ``starts_with``, limits the returned flatpages to
    those beginning with a particular base URL. This argument can be a variable
    or a string, as it resolves from the template context.

    Syntax::

        {% get_flatpages ['url_starts_with'] [for user] as context_name %}

    Example usage::

        {% get_flatpages as flatpages %}
        {% get_flatpages for someuser as flatpages %}
        {% get_flatpages '/about/' as about_pages %}
        {% get_flatpages prefix as about_pages %}
        {% get_flatpages '/about/' for someuser as about_pages %}
    """
    bits = token.split_contents()
    syntax_message = "%(tag_name)s expects a syntax of %(tag_name)s ['url_starts_with'] [for user] as context_name" % {'tag_name': bits[0]}
    if 3 <= len(bits) <= 6:
        if len(bits) % 2 == 0:
            prefix = bits[1]
        else:
            prefix = None
        if bits[-2] != 'as':
            raise template.TemplateSyntaxError(syntax_message)
        context_name = bits[-1]
        if len(bits) >= 5:
            if bits[-4] != 'for':
                raise template.TemplateSyntaxError(syntax_message)
            user = bits[-3]
        else:
            user = None
        return FlatpageNode(context_name, starts_with=prefix, user=user)
    else:
        raise template.TemplateSyntaxError(syntax_message)