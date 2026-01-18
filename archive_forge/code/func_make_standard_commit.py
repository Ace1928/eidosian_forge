import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def make_standard_commit(self, branch_nick, **kwargs):
    wt = self.make_branch_and_tree('.')
    wt.lock_write()
    self.addCleanup(wt.unlock)
    self.build_tree(['a'])
    wt.add(['a'])
    wt.branch.nick = branch_nick
    kwargs.setdefault('committer', 'Lorem Ipsum <test@example.com>')
    kwargs.setdefault('authors', ['John Doe <jdoe@example.com>'])
    self.wt_commit(wt, 'add a', **kwargs)
    return wt