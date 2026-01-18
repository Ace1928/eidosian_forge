from git.config import GitConfigParser, SectionConstraint
from git.util import join_path
from git.exc import GitCommandError
from .symbolic import SymbolicReference
from .reference import Reference
from typing import Any, Sequence, Union, TYPE_CHECKING
from git.types import PathLike, Commit_ish
def checkout(self, force: bool=False, **kwargs: Any) -> Union['HEAD', 'Head']:
    """Check out this head by setting the HEAD to this reference, by updating the
        index to reflect the tree we point to and by updating the working tree to
        reflect the latest index.

        The command will fail if changed working tree files would be overwritten.

        :param force:
            If True, changes to the index and the working tree will be discarded.
            If False, :class:`~git.exc.GitCommandError` will be raised in that
            situation.

        :param kwargs:
            Additional keyword arguments to be passed to git checkout, e.g.
            ``b="new_branch"`` to create a new branch at the given spot.

        :return:
            The active branch after the checkout operation, usually self unless
            a new branch has been created.
            If there is no active branch, as the HEAD is now detached, the HEAD
            reference will be returned instead.

        :note:
            By default it is only allowed to checkout heads - everything else
            will leave the HEAD detached which is allowed and possible, but remains
            a special state that some tools might not be able to handle.
        """
    kwargs['f'] = force
    if kwargs['f'] is False:
        kwargs.pop('f')
    self.repo.git.checkout(self, **kwargs)
    if self.repo.head.is_detached:
        return self.repo.head
    else:
        return self.repo.active_branch