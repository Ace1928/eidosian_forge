from ... import errors
from ... import transport as _mod_transport
from ...bzr.bzrdir import BzrFormat
from ...commands import plugin_cmds
from ...i18n import load_plugin_translations
from ...hooks import install_lazy_named_hook
def show_rebase_summary(params):
    if getattr(params.new_tree, '_format', None) is None:
        return
    features = getattr(params.new_tree._format, 'features', None)
    if features is None:
        return
    if 'rebase-v1' not in features:
        return
    from .rebase import RebaseState1, rebase_todo
    state = RebaseState1(params.new_tree)
    try:
        replace_map = state.read_plan()[1]
    except _mod_transport.NoSuchFile:
        return
    todo = list(rebase_todo(params.new_tree.branch.repository, replace_map))
    params.to_file.write('Rebase in progress. (%d revisions left)\n' % len(todo))