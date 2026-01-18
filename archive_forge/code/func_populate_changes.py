from __future__ import annotations
import typing as t
from .util import (
from .io import (
from .diff import (
def populate_changes(self, diff: t.Optional[list[str]]) -> None:
    """Populate the changeset using the given diff."""
    patches = parse_diff(diff)
    patches: list[FileDiff] = sorted(patches, key=lambda k: k.new.path)
    self.changes = dict(((patch.new.path, tuple(patch.new.ranges)) for patch in patches))
    renames = [patch.old.path for patch in patches if patch.old.path != patch.new.path and patch.old.exists and patch.new.exists]
    deletes = [patch.old.path for patch in patches if not patch.new.exists]
    for path in renames + deletes:
        if path in self.changes:
            continue
        self.changes[path] = ((0, 0),)