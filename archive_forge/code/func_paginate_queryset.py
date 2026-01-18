from django.core.exceptions import ImproperlyConfigured
from django.core.paginator import InvalidPage, Paginator
from django.db.models import QuerySet
from django.http import Http404
from django.utils.translation import gettext as _
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
def paginate_queryset(self, queryset, page_size):
    """Paginate the queryset, if needed."""
    paginator = self.get_paginator(queryset, page_size, orphans=self.get_paginate_orphans(), allow_empty_first_page=self.get_allow_empty())
    page_kwarg = self.page_kwarg
    page = self.kwargs.get(page_kwarg) or self.request.GET.get(page_kwarg) or 1
    try:
        page_number = int(page)
    except ValueError:
        if page == 'last':
            page_number = paginator.num_pages
        else:
            raise Http404(_('Page is not “last”, nor can it be converted to an int.'))
    try:
        page = paginator.page(page_number)
        return (paginator, page, page.object_list, page.has_other_pages())
    except InvalidPage as e:
        raise Http404(_('Invalid page (%(page_number)s): %(message)s') % {'page_number': page_number, 'message': str(e)})