import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def set_reference(self, ref: Union[Commit_ish, 'SymbolicReference', str], logmsg: Union[str, None]=None) -> 'SymbolicReference':
    """Set ourselves to the given ref. It will stay a symbol if the ref is a Reference.
        Otherwise an Object, given as Object instance or refspec, is assumed and if valid,
        will be set which effectively detaches the reference if it was a purely
        symbolic one.

        :param ref:
            A :class:`SymbolicReference` instance, an :class:`~git.objects.base.Object`
            instance, or a refspec string. Only if the ref is a
            :class:`SymbolicReference` instance, we will point to it. Everything else is
            dereferenced to obtain the actual object.

        :param logmsg: If set to a string, the message will be used in the reflog.
            Otherwise, a reflog entry is not written for the changed reference.
            The previous commit of the entry will be the commit we point to now.

            See also: :meth:`log_append`

        :return: self

        :note: This symbolic reference will not be dereferenced. For that, see
            :meth:`set_object`.
        """
    write_value = None
    obj = None
    if isinstance(ref, SymbolicReference):
        write_value = 'ref: %s' % ref.path
    elif isinstance(ref, Object):
        obj = ref
        write_value = ref.hexsha
    elif isinstance(ref, str):
        try:
            obj = self.repo.rev_parse(ref + '^{}')
            write_value = obj.hexsha
        except (BadObject, BadName) as e:
            raise ValueError('Could not extract object from %s' % ref) from e
    else:
        raise ValueError('Unrecognized Value: %r' % ref)
    if obj is not None and self._points_to_commits_only and (obj.type != Commit.type):
        raise TypeError('Require commit, got %r' % obj)
    oldbinsha: bytes = b''
    if logmsg is not None:
        try:
            oldbinsha = self.commit.binsha
        except ValueError:
            oldbinsha = Commit.NULL_BIN_SHA
    fpath = self.abspath
    assure_directory_exists(fpath, is_file=True)
    lfd = LockedFD(fpath)
    fd = lfd.open(write=True, stream=True)
    try:
        fd.write(write_value.encode('utf-8') + b'\n')
        lfd.commit()
    except BaseException:
        lfd.rollback()
        raise
    if logmsg is not None:
        self.log_append(oldbinsha, logmsg)
    return self