import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def emit_tags(self):
    for tag, revid in self.branch.tags.get_tag_dict().items():
        try:
            mark = self.revid_to_mark[revid]
        except KeyError:
            self.warning('not creating tag %r pointing to non-existent revision %s' % (tag, revid))
        else:
            git_ref = b'refs/tags/%s' % tag.encode('utf-8')
            if self.plain_format and (not check_ref_format(git_ref)):
                if self.rewrite_tags:
                    new_ref = sanitize_ref_name_for_git(git_ref)
                    self.warning('tag %r is exported as %r to be valid in git.', git_ref, new_ref)
                    git_ref = new_ref
                else:
                    self.warning('not creating tag %r as its name would not be valid in git.', git_ref)
                    continue
            self.print_cmd(commands.ResetCommand(git_ref, b':%s' % mark))