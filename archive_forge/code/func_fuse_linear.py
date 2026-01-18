from __future__ import annotations
import math
import numbers
from collections.abc import Iterable
from enum import Enum
from typing import Any
from dask import config, core, utils
from dask.base import normalize_token, tokenize
from dask.core import (
from dask.typing import Graph, Key
def fuse_linear(dsk, keys=None, dependencies=None, rename_keys=True):
    """Return new dask graph with linear sequence of tasks fused together.

    If specified, the keys in ``keys`` keyword argument are *not* fused.
    Supply ``dependencies`` from output of ``cull`` if available to avoid
    recomputing dependencies.

    **This function is mostly superseded by ``fuse``**

    Parameters
    ----------
    dsk: dict
    keys: list
    dependencies: dict, optional
        {key: [list-of-keys]}.  Must be a list to provide count of each key
        This optional input often comes from ``cull``
    rename_keys: bool or func, optional
        Whether to rename fused keys with ``default_fused_linear_keys_renamer``
        or not.  Renaming fused keys can keep the graph more understandable
        and comprehensive, but it comes at the cost of additional processing.
        If False, then the top-most key will be used.  For advanced usage, a
        func is also accepted, ``new_key = rename_keys(fused_key_list)``.

    Examples
    --------
    >>> def inc(x):
    ...     return x + 1

    >>> def add(x, y):
    ...     return x + y

    >>> d = {'a': 1, 'b': (inc, 'a'), 'c': (inc, 'b')}
    >>> dsk, dependencies = fuse(d)
    >>> dsk # doctest: +SKIP
    {'a-b-c': (inc, (inc, 1)), 'c': 'a-b-c'}
    >>> dsk, dependencies = fuse(d, rename_keys=False)
    >>> dsk # doctest: +ELLIPSIS
    {'c': (<function inc at ...>, (<function inc at ...>, 1))}
    >>> dsk, dependencies = fuse(d, keys=['b'], rename_keys=False)
    >>> dsk  # doctest: +ELLIPSIS
    {'b': (<function inc at ...>, 1), 'c': (<function inc at ...>, 'b')}

    Returns
    -------
    dsk: output graph with keys fused
    dependencies: dict mapping dependencies after fusion.  Useful side effect
        to accelerate other downstream optimizations.
    """
    if keys is not None and (not isinstance(keys, set)):
        if not isinstance(keys, list):
            keys = [keys]
        keys = set(flatten(keys))
    if dependencies is None:
        dependencies = {k: get_dependencies(dsk, k, as_list=True) for k in dsk}
    child2parent = {}
    unfusible = set()
    for parent in dsk:
        deps = dependencies[parent]
        has_many_children = len(deps) > 1
        for child in deps:
            if keys is not None and child in keys:
                unfusible.add(child)
            elif child in child2parent:
                del child2parent[child]
                unfusible.add(child)
            elif has_many_children:
                unfusible.add(child)
            elif child not in unfusible:
                child2parent[child] = parent
    chains = []
    parent2child = dict(map(reversed, child2parent.items()))
    while child2parent:
        child, parent = child2parent.popitem()
        chain = [child, parent]
        while parent in child2parent:
            parent = child2parent.pop(parent)
            del parent2child[parent]
            chain.append(parent)
        chain.reverse()
        while child in parent2child:
            child = parent2child.pop(child)
            del child2parent[child]
            chain.append(child)
        chains.append(chain)
    dependencies = {k: set(v) for k, v in dependencies.items()}
    if rename_keys is True:
        key_renamer = default_fused_linear_keys_renamer
    elif rename_keys is False:
        key_renamer = None
    else:
        key_renamer = rename_keys
    rv = {}
    fused = set()
    aliases = set()
    is_renamed = False
    for chain in chains:
        if key_renamer is not None:
            new_key = key_renamer(chain)
            is_renamed = new_key is not None and new_key not in dsk and (new_key not in rv)
        child = chain.pop()
        val = dsk[child]
        while chain:
            parent = chain.pop()
            dependencies[parent].update(dependencies.pop(child))
            dependencies[parent].remove(child)
            val = subs(dsk[parent], child, val)
            fused.add(child)
            child = parent
        fused.add(child)
        if is_renamed:
            rv[new_key] = val
            rv[child] = new_key
            dependencies[new_key] = dependencies[child]
            dependencies[child] = {new_key}
            aliases.add(child)
        else:
            rv[child] = val
    for key, val in dsk.items():
        if key not in fused:
            rv[key] = val
    if aliases:
        for key, deps in dependencies.items():
            for old_key in deps & aliases:
                new_key = rv[old_key]
                deps.remove(old_key)
                deps.add(new_key)
                rv[key] = subs(rv[key], old_key, new_key)
        if keys is not None:
            for key in aliases - keys:
                del rv[key]
                del dependencies[key]
    return (rv, dependencies)