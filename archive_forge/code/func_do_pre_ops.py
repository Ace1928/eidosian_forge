from oslo_log import log as logging
from heat.engine import resources
def do_pre_ops(cnxt, stack, current_stack=None, action=None):
    """Call available pre-op methods sequentially.

    In order determined with get_ordinal(), with parameters context, stack,
    current_stack, action.

    On failure of any pre_op method, will call post-op methods corresponding
    to successful calls of pre-op methods.
    """
    cinstances = get_plug_point_class_instances()
    if action is None:
        action = stack.action
    failure, failure_exception_message, success_count = _do_ops(cinstances, 'do_pre_op', cnxt, stack, current_stack, action, None)
    if failure:
        cinstances = cinstances[0:success_count]
        _do_ops(cinstances, 'do_post_op', cnxt, stack, current_stack, action, True)
        raise Exception(failure_exception_message)