from copy import copy
from ..osutils import contains_linebreaks, contains_whitespace, sha_strings
from ..tree import Tree
@classmethod
def from_revision_tree(cls, tree):
    """Produce a new testament from a revision tree."""
    rev = tree._repository.get_revision(tree.get_revision_id())
    return cls(rev, tree)