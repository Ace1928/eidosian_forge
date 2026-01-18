import contextlib
from .branch import Branch
from .errors import CommandError, NoSuchRevision
from .trace import note
def iter_ghosts(self):
    """Find all ancestors that aren't stored in this branch."""
    seen = set()
    lines = [self.this_branch.last_revision()]
    if lines[0] is None:
        return
    while len(lines) > 0:
        new_lines = []
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            try:
                revision = self.this_branch.repository.get_revision(line)
                new_lines.extend(revision.parent_ids)
            except NoSuchRevision:
                yield line
        lines = new_lines