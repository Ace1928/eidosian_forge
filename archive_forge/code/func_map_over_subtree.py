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
def map_over_subtree(self, func: Callable, *args: Iterable[Any], **kwargs: Any) -> DataTree | tuple[DataTree]:
    """
        Apply a function to every dataset in this subtree, returning a new tree which stores the results.

        The function will be applied to any dataset stored in this node, as well as any dataset stored in any of the
        descendant nodes. The returned tree will have the same structure as the original subtree.

        func needs to return a Dataset in order to rebuild the subtree.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.ds, *args, **kwargs) -> Dataset`.

            Function will not be applied to any nodes without datasets.
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.

        Returns
        -------
        subtrees : DataTree, tuple of DataTrees
            One or more subtrees containing results from applying ``func`` to the data at each node.
        """
    return map_over_subtree(func)(self, *args, **kwargs)