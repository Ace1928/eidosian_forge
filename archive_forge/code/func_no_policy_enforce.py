import functools
from webob import exc
from heat.common.i18n import _
from heat.common import identifier
def no_policy_enforce(handler):
    """Decorator that does *not* enforce policies.

    Checks the path matches the request context.

    This is a handler method decorator.
    """

    @functools.wraps(handler)
    def handle_stack_method(controller, req, tenant_id, **kwargs):
        if req.context.tenant_id != tenant_id and (not req.context.is_admin):
            raise exc.HTTPForbidden()
        return handler(controller, req, **kwargs)
    return handle_stack_method