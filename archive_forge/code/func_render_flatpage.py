from django.conf import settings
from django.contrib.flatpages.models import FlatPage
from django.contrib.sites.shortcuts import get_current_site
from django.http import Http404, HttpResponse, HttpResponsePermanentRedirect
from django.shortcuts import get_object_or_404
from django.template import loader
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_protect
@csrf_protect
def render_flatpage(request, f):
    """
    Internal interface to the flat page view.
    """
    if f.registration_required and (not request.user.is_authenticated):
        from django.contrib.auth.views import redirect_to_login
        return redirect_to_login(request.path)
    if f.template_name:
        template = loader.select_template((f.template_name, DEFAULT_TEMPLATE))
    else:
        template = loader.get_template(DEFAULT_TEMPLATE)
    f.title = mark_safe(f.title)
    f.content = mark_safe(f.content)
    return HttpResponse(template.render({'flatpage': f}, request))