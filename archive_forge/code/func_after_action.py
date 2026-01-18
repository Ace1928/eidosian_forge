from inspect import getmembers, isclass, isfunction
from .util import _cfg, getargspec
def after_action(action_type, action):
    """
    If utilizing the :mod:`pecan.hooks` ``TransactionHook``, allows you
    to flag a controller method to perform a callable action after the
    action_type is successfully issued.

    :param action: The callable to call after the commit is successfully
    issued.  """
    if action_type not in ('commit', 'rollback'):
        raise Exception('action_type (%s) is not valid' % action_type)

    def deco(func):
        _cfg(func).setdefault('after_%s' % action_type, []).append(action)
        return func
    return deco