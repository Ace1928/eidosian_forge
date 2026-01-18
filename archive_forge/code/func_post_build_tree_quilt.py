from ... import trace
from ...errors import BzrError
from ...hooks import install_lazy_named_hook
from ...config import Option, bool_from_store, option_registry
def post_build_tree_quilt(tree):
    config = tree.get_config_stack()
    policy = config.get('quilt.tree_policy')
    if policy is None:
        return
    from .merge import post_process_quilt_patches
    from .wrapper import QuiltNotInstalled
    try:
        post_process_quilt_patches(tree, [], policy)
    except QuiltNotInstalled:
        trace.warning('quilt not installed; not touching patches')