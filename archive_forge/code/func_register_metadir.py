from typing import TYPE_CHECKING
from .. import config, controldir, errors, pyutils, registry
from .. import transport as _mod_transport
from ..branch import format_registry as branch_format_registry
from ..repository import format_registry as repository_format_registry
from ..workingtree import format_registry as workingtree_format_registry
def register_metadir(registry, key, repository_format, help, native=True, deprecated=False, branch_format=None, tree_format=None, hidden=False, experimental=False, bzrdir_format=None):
    """Register a metadir subformat.

    These all use a meta bzrdir, but can be parameterized by the
    Repository/Branch/WorkingTreeformats.

    :param repository_format: The fully-qualified repository format class
        name as a string.
    :param branch_format: Fully-qualified branch format class name as
        a string.
    :param tree_format: Fully-qualified tree format class name as
        a string.
    """
    if bzrdir_format is None:
        bzrdir_format = 'breezy.bzr.bzrdir.BzrDirMetaFormat1'

    def _load(full_name):
        mod_name, factory_name = full_name.rsplit('.', 1)
        try:
            factory = pyutils.get_named_object(mod_name, factory_name)
        except ImportError as e:
            raise ImportError('failed to load {}: {}'.format(full_name, e))
        except AttributeError:
            raise AttributeError('no factory %s in module %r' % (full_name, sys.modules[mod_name]))
        return factory()

    def helper():
        bd = _load(bzrdir_format)
        if branch_format is not None:
            bd.set_branch_format(_load(branch_format))
        if tree_format is not None:
            bd.workingtree_format = _load(tree_format)
        if repository_format is not None:
            bd.repository_format = _load(repository_format)
        return bd
    registry.register(key, helper, help, native, deprecated, hidden, experimental)