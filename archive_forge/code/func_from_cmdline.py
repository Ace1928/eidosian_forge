import contextlib
from .branch import Branch
from .errors import CommandError, NoSuchRevision
from .trace import note
@classmethod
def from_cmdline(cls, other):
    this_branch = Branch.open_containing('.')[0]
    if other is None:
        other = this_branch.get_parent()
        if other is None:
            raise CommandError('No branch specified and no location saved.')
        else:
            note('Using saved location %s.', other)
    other_branch = Branch.open_containing(other)[0]
    return cls(this_branch, other_branch)