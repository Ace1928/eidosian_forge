from ... import version_info  # noqa: F401
from ...config import option_registry
from ...hooks import install_lazy_named_hook
def branch_post_change_hook(params):
    """This is the post_change_branch_tip hook."""
    from . import emailer
    emailer.EmailSender(params.branch, params.new_revid, params.branch.get_config_stack(), local_branch=None, op='change').send_maybe()