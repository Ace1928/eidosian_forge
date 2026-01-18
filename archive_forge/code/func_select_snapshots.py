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
def select_snapshots(self, vf):
    """Determine which versions to add as snapshots"""
    build_ancestors = {}
    snapshots = set()
    for version_id in topo_iter(vf):
        potential_build_ancestors = set(vf.get_parents(version_id))
        parents = vf.get_parents(version_id)
        if len(parents) == 0:
            snapshots.add(version_id)
            build_ancestors[version_id] = set()
        else:
            for parent in vf.get_parents(version_id):
                potential_build_ancestors.update(build_ancestors[parent])
            if len(potential_build_ancestors) > self.snapshot_interval:
                snapshots.add(version_id)
                build_ancestors[version_id] = set()
            else:
                build_ancestors[version_id] = potential_build_ancestors
    return snapshots