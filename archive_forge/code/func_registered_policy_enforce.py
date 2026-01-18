import functools
from webob import exc
from heat.common.i18n import _
from heat.common import identifier
def registered_policy_enforce(handler):
    """Decorator that enforces policies.

    Checks the path matches the request context and enforce policy defined in
    policies.

    This is a handler method decorator.
    """

    @functools.wraps(handler)
    def handle_stack_method(controller, req, tenant_id, **kwargs):
        if req.context.is_admin and req.context.project_id:
            tenant_id = req.context.tenant_id
        _target = {'project_id': tenant_id}
        if req.context.tenant_id != tenant_id:
            raise exc.HTTPForbidden()
        allowed = req.context.policy.enforce(context=req.context, action=handler.__name__, scope=controller.REQUEST_SCOPE, target=_target, is_registered_policy=True)
        if not allowed:
            raise exc.HTTPForbidden()
        return handler(controller, req, **kwargs)
    return handle_stack_method