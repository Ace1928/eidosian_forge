import contextlib
import logging
import re
from git.cmd import Git, handle_process_output
from git.compat import defenc, force_text
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import GitCommandError
from git.refs import Head, Reference, RemoteReference, SymbolicReference, TagReference
from git.util import (
from typing import (
from git.types import PathLike, Literal, Commit_ish
@property
def remote_ref(self) -> Union[RemoteReference, TagReference]:
    """
        :return:
            Remote :class:`~git.refs.reference.Reference` or
            :class:`~git.refs.tag.TagReference` in the local repository corresponding to
            the :attr:`remote_ref_string` kept in this instance.
        """
    if self.remote_ref_string.startswith('refs/tags'):
        return TagReference(self._remote.repo, self.remote_ref_string)
    elif self.remote_ref_string.startswith('refs/heads'):
        remote_ref = Reference(self._remote.repo, self.remote_ref_string)
        return RemoteReference(self._remote.repo, 'refs/remotes/%s/%s' % (str(self._remote), remote_ref.name))
    else:
        raise ValueError('Could not handle remote ref: %r' % self.remote_ref_string)