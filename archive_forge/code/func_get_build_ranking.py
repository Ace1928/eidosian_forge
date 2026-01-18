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
def get_build_ranking(self):
    """Return revisions sorted by how much they reduce build complexity"""
    could_avoid = {}
    referenced_by = {}
    for version_id in topo_iter(self):
        could_avoid[version_id] = set()
        if version_id not in self._snapshots:
            for parent_id in self._parents[version_id]:
                could_avoid[version_id].update(could_avoid[parent_id])
            could_avoid[version_id].update(self._parents)
            could_avoid[version_id].discard(version_id)
        for avoid_id in could_avoid[version_id]:
            referenced_by.setdefault(avoid_id, set()).add(version_id)
    available_versions = list(self.versions())
    ranking = []
    while len(available_versions) > 0:
        available_versions.sort(key=lambda x: len(could_avoid[x]) * len(referenced_by.get(x, [])))
        selected = available_versions.pop()
        ranking.append(selected)
        for version_id in referenced_by[selected]:
            could_avoid[version_id].difference_update(could_avoid[selected])
        for version_id in could_avoid[selected]:
            referenced_by[version_id].difference_update(referenced_by[selected])
    return ranking