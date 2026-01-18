from ... import trace
from ...errors import BzrError
from ...hooks import install_lazy_named_hook
from ...config import Option, bool_from_store, option_registry
def policy_from_store(s):
    if s not in ('applied', 'unapplied'):
        raise ValueError('Invalid quilt.commit_policy: %s' % s)
    return s