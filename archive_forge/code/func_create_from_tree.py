import datetime
import re
from subprocess import Popen, PIPE
from gitdb import IStream
from git.util import hex_to_bin, Actor, Stats, finalize_process
from git.diff import Diffable
from git.cmd import Git
from .tree import Tree
from . import base
from .util import (
from time import time, daylight, altzone, timezone, localtime
import os
from io import BytesIO
import logging
from collections import defaultdict
from typing import (
from git.types import PathLike, Literal
@classmethod
def create_from_tree(cls, repo: 'Repo', tree: Union[Tree, str], message: str, parent_commits: Union[None, List['Commit']]=None, head: bool=False, author: Union[None, Actor]=None, committer: Union[None, Actor]=None, author_date: Union[None, str, datetime.datetime]=None, commit_date: Union[None, str, datetime.datetime]=None) -> 'Commit':
    """Commit the given tree, creating a commit object.

        :param repo: Repo object the commit should be part of
        :param tree: Tree object or hex or bin sha. The tree of the new commit.
        :param message: Commit message. It may be an empty string if no message is
            provided. It will be converted to a string, in any case.
        :param parent_commits:
            Optional :class:`Commit` objects to use as parents for the new commit.
            If empty list, the commit will have no parents at all and become
            a root commit.
            If None, the current head commit will be the parent of the
            new commit object.
        :param head:
            If True, the HEAD will be advanced to the new commit automatically.
            Otherwise the HEAD will remain pointing on the previous commit. This could
            lead to undesired results when diffing files.
        :param author: The name of the author, optional. If unset, the repository
            configuration is used to obtain this value.
        :param committer: The name of the committer, optional. If unset, the
            repository configuration is used to obtain this value.
        :param author_date: The timestamp for the author field.
        :param commit_date: The timestamp for the committer field.

        :return: Commit object representing the new commit.

        :note:
            Additional information about the committer and Author are taken from the
            environment or from the git configuration, see git-commit-tree for
            more information.
        """
    if parent_commits is None:
        try:
            parent_commits = [repo.head.commit]
        except ValueError:
            parent_commits = []
    else:
        for p in parent_commits:
            if not isinstance(p, cls):
                raise ValueError(f"Parent commit '{p!r}' must be of type {cls}")
    cr = repo.config_reader()
    env = os.environ
    committer = committer or Actor.committer(cr)
    author = author or Actor.author(cr)
    unix_time = int(time())
    is_dst = daylight and localtime().tm_isdst > 0
    offset = altzone if is_dst else timezone
    author_date_str = env.get(cls.env_author_date, '')
    if author_date:
        author_time, author_offset = parse_date(author_date)
    elif author_date_str:
        author_time, author_offset = parse_date(author_date_str)
    else:
        author_time, author_offset = (unix_time, offset)
    committer_date_str = env.get(cls.env_committer_date, '')
    if commit_date:
        committer_time, committer_offset = parse_date(commit_date)
    elif committer_date_str:
        committer_time, committer_offset = parse_date(committer_date_str)
    else:
        committer_time, committer_offset = (unix_time, offset)
    enc_section, enc_option = cls.conf_encoding.split('.')
    conf_encoding = cr.get_value(enc_section, enc_option, cls.default_encoding)
    if not isinstance(conf_encoding, str):
        raise TypeError('conf_encoding could not be coerced to str')
    if isinstance(tree, str):
        tree = repo.tree(tree)
    new_commit = cls(repo, cls.NULL_BIN_SHA, tree, author, author_time, author_offset, committer, committer_time, committer_offset, message, parent_commits, conf_encoding)
    new_commit.binsha = cls._calculate_sha_(repo, new_commit)
    if head:
        import git.refs
        try:
            repo.head.set_commit(new_commit, logmsg=message)
        except ValueError:
            master = git.refs.Head.create(repo, repo.head.ref, new_commit, logmsg='commit (initial): %s' % message)
            repo.head.set_reference(master, logmsg='commit: Switching to %s' % master)
    return new_commit