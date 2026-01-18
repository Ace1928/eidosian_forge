from ... import trace
from ...errors import BzrError
from ...hooks import install_lazy_named_hook
from ...config import Option, bool_from_store, option_registry
def post_merge_quilt_cleanup(merger):
    import shutil
    for dir in getattr(merger, '_quilt_tempdirs', []):
        shutil.rmtree(dir)
    config = merger.working_tree.get_config_stack()
    policy = config.get('quilt.tree_policy')
    if policy is None:
        return
    from .merge import post_process_quilt_patches
    post_process_quilt_patches(merger.working_tree, getattr(merger, '_old_quilt_series', []), policy)