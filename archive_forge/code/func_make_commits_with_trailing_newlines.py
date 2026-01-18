import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def make_commits_with_trailing_newlines(self, wt):
    """Helper method for LogFormatter tests"""
    b = wt.branch
    b.nick = 'test'
    self.build_tree_contents([('a', b'hello moto\n')])
    self.wt_commit(wt, 'simple log message', rev_id=b'a1')
    self.build_tree_contents([('b', b'goodbye\n')])
    wt.add('b')
    self.wt_commit(wt, 'multiline\nlog\nmessage\n', rev_id=b'a2')
    self.build_tree_contents([('c', b'just another manic monday\n')])
    wt.add('c')
    self.wt_commit(wt, 'single line with trailing newline\n', rev_id=b'a3')
    return b