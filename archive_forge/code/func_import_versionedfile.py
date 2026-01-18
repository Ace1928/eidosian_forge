import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
def import_versionedfile(self, vf, snapshots, no_cache=True, single_parent=False, verify=False):
    """Import all revisions of a versionedfile

        :param vf: The versionedfile to import
        :param snapshots: If provided, the revisions to make snapshots of.
            Otherwise, this will be auto-determined
        :param no_cache: If true, clear the cache after every add.
        :param single_parent: If true, omit all but one parent text, (but
            retain parent metadata).
        """
    if not (no_cache or not verify):
        raise ValueError()
    revisions = set(vf.versions())
    total = len(revisions)
    with ui.ui_factory.nested_progress_bar() as pb:
        while len(revisions) > 0:
            added = set()
            for revision in revisions:
                parents = vf.get_parents(revision)
                if [p for p in parents if p not in self._parents] != []:
                    continue
                lines = [a + b' ' + l for a, l in vf.annotate(revision)]
                if snapshots is None:
                    force_snapshot = None
                else:
                    force_snapshot = revision in snapshots
                self.add_version(lines, revision, parents, force_snapshot, single_parent)
                added.add(revision)
                if no_cache:
                    self.clear_cache()
                    vf.clear_cache()
                    if verify:
                        if not lines == self.get_line_list([revision])[0]:
                            raise AssertionError()
                        self.clear_cache()
                pb.update(gettext('Importing revisions'), total - len(revisions) + len(added), total)
            revisions = [r for r in revisions if r not in added]