import stat
from typing import Dict, Tuple
from fastimport import commands, parser, processor
from fastimport import errors as fastimport_errors
from .index import commit_tree
from .object_store import iter_tree_contents
from .objects import ZERO_SHA, Blob, Commit, Tag
def import_stream(self, stream):
    p = parser.ImportParser(stream)
    self.process(p.iter_commands)
    return self.markers