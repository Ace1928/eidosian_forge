from ... import version_info  # noqa: F401
from ... import commands, config, hooks
def install_auto_upload_hook():
    hooks.install_lazy_named_hook('breezy.branch', 'Branch.hooks', 'post_change_branch_tip', auto_upload_hook, 'Auto upload code from a branch when it is changed.')