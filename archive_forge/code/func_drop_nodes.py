from __future__ import annotations
import copy
import itertools
from collections.abc import Hashable, Iterable, Iterator, Mapping, MutableMapping
from html import escape
from typing import (
from xarray.core import utils
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, DataVariables
from xarray.core.indexes import Index, Indexes
from xarray.core.merge import dataset_update_method
from xarray.core.options import OPTIONS as XR_OPTS
from xarray.core.treenode import NamedNode, NodePath, Tree
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.datatree_.datatree.common import TreeAttrAccessMixin
from xarray.datatree_.datatree.formatting import datatree_repr
from xarray.datatree_.datatree.formatting_html import (
from xarray.datatree_.datatree.mapping import (
from xarray.datatree_.datatree.ops import (
from xarray.datatree_.datatree.render import RenderTree
def drop_nodes(self: DataTree, names: str | Iterable[str], *, errors: ErrorOptions='raise') -> DataTree:
    """
        Drop child nodes from this node.

        Parameters
        ----------
        names : str or iterable of str
            Name(s) of nodes to drop.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a KeyError if any of the node names
            passed are not present as children of this node. If 'ignore',
            any given names that are present are dropped and no error is raised.

        Returns
        -------
        dropped : DataTree
            A copy of the node with the specified children dropped.
        """
    if isinstance(names, str) or not isinstance(names, Iterable):
        names = {names}
    else:
        names = set(names)
    if errors == 'raise':
        extra = names - set(self.children)
        if extra:
            raise KeyError(f'Cannot drop all nodes - nodes {extra} not present')
    children_to_keep = {name: child for name, child in self.children.items() if name not in names}
    return self._replace(children=children_to_keep)