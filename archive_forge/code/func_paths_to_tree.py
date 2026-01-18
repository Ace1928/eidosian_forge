from __future__ import annotations
import abc
import collections
import os
import typing as t
from ...util import (
from .. import (
def paths_to_tree(paths: list[str]) -> tuple[dict[str, t.Any], list[str]]:
    """Return a filesystem tree from the given list of paths."""
    tree: tuple[dict[str, t.Any], list[str]] = ({}, [])
    for path in paths:
        parts = path.split(os.path.sep)
        root = tree
        for part in parts[:-1]:
            if part not in root[0]:
                root[0][part] = ({}, [])
            root = root[0][part]
        root[1].append(path)
    return tree