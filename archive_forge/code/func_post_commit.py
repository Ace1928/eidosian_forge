from ... import version_info  # noqa: F401
from ...config import option_registry
from ...hooks import install_lazy_named_hook
def post_commit(branch, revision_id):
    """This is the post_commit hook that should get run after commit."""
    from . import emailer
    emailer.EmailSender(branch, revision_id, branch.get_config_stack()).send_maybe()