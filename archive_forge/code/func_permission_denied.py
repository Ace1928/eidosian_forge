from urllib.parse import quote
from django.http import (
from django.template import Context, Engine, TemplateDoesNotExist, loader
from django.views.decorators.csrf import requires_csrf_token
@requires_csrf_token
def permission_denied(request, exception, template_name=ERROR_403_TEMPLATE_NAME):
    """
    Permission denied (403) handler.

    Templates: :template:`403.html`
    Context:
        exception
            The message from the exception which triggered the 403 (if one was
            supplied).

    If the template does not exist, an Http403 response containing the text
    "403 Forbidden" (as per RFC 9110 Section 15.5.4) will be returned.
    """
    try:
        template = loader.get_template(template_name)
    except TemplateDoesNotExist:
        if template_name != ERROR_403_TEMPLATE_NAME:
            raise
        return HttpResponseForbidden(ERROR_PAGE_TEMPLATE % {'title': '403 Forbidden', 'details': ''})
    return HttpResponseForbidden(template.render(request=request, context={'exception': str(exception)}))