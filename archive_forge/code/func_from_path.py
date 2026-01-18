import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
@classmethod
def from_path(cls: Type[T_References], repo: 'Repo', path: PathLike) -> T_References:
    """
        Make a symbolic reference from a path.

        :param path: Full ``.git``-directory-relative path name to the Reference to
            instantiate.

        :note: Use :meth:`to_full_path` if you only have a partial path of a known
            Reference type.

        :return:
            Instance of type :class:`~git.refs.reference.Reference`,
            :class:`~git.refs.head.Head`, or :class:`~git.refs.tag.Tag`, depending on
            the given path.
        """
    if not path:
        raise ValueError('Cannot create Reference from %r' % path)
    from . import HEAD, Head, RemoteReference, TagReference, Reference
    for ref_type in (HEAD, Head, RemoteReference, TagReference, Reference, SymbolicReference):
        try:
            instance: T_References
            instance = ref_type(repo, path)
            if instance.__class__ is SymbolicReference and instance.is_detached:
                raise ValueError('SymbolicRef was detached, we drop it')
            else:
                return instance
        except ValueError:
            pass
    raise ValueError('Could not find reference type suitable to handle path %r' % path)