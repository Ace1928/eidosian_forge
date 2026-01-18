import inspect
from django.conf import settings
from .. import Error, Tags, Warning, register
@register(Tags.security)
def check_csrf_failure_view(app_configs, **kwargs):
    from django.middleware.csrf import _get_failure_view
    errors = []
    try:
        view = _get_failure_view()
    except ImportError:
        msg = "The CSRF failure view '%s' could not be imported." % settings.CSRF_FAILURE_VIEW
        errors.append(Error(msg, id='security.E102'))
    else:
        try:
            inspect.signature(view).bind(None, reason=None)
        except TypeError:
            msg = "The CSRF failure view '%s' does not take the correct number of arguments." % settings.CSRF_FAILURE_VIEW
            errors.append(Error(msg, id='security.E101'))
    return errors