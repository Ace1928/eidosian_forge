from django.http import (
from django.template import loader
from django.urls import NoReverseMatch, reverse
from django.utils.functional import Promise
def resolve_url(to, *args, **kwargs):
    """
    Return a URL appropriate for the arguments passed.

    The arguments could be:

        * A model: the model's `get_absolute_url()` function will be called.

        * A view name, possibly with arguments: `urls.reverse()` will be used
          to reverse-resolve the name.

        * A URL, which will be returned as-is.
    """
    if hasattr(to, 'get_absolute_url'):
        return to.get_absolute_url()
    if isinstance(to, Promise):
        to = str(to)
    if isinstance(to, str) and to.startswith(('./', '../')):
        return to
    try:
        return reverse(to, args=args, kwargs=kwargs)
    except NoReverseMatch:
        if callable(to):
            raise
        if '/' not in to and '.' not in to:
            raise
    return to