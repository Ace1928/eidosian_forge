from contextlib import contextmanager
from copy import copy
def make_context(context, request=None, **kwargs):
    """
    Create a suitable Context from a plain dict and optionally an HttpRequest.
    """
    if context is not None and (not isinstance(context, dict)):
        raise TypeError('context must be a dict rather than %s.' % context.__class__.__name__)
    if request is None:
        context = Context(context, **kwargs)
    else:
        original_context = context
        context = RequestContext(request, **kwargs)
        if original_context:
            context.push(original_context)
    return context