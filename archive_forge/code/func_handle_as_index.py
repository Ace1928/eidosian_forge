import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@staticmethod
def handle_as_index(result_cols, result_index_names, internal_by_cols, by_cols_dtypes=None, by_length=None, selection=None, partition_idx=0, drop=True, method=None):
    """
        Compute hints to process ``as_index=False`` parameter for the GroupBy result.

        This function resolves naming conflicts of the index levels to insert and the column labels
        for the GroupBy result. The logic of this function assumes that the initial GroupBy result
        was computed as ``as_index=True``.

        Parameters
        ----------
        result_cols : pandas.Index
            Columns of the GroupBy result.
        result_index_names : list-like
            Index names of the GroupBy result.
        internal_by_cols : list-like
            Internal 'by' columns.
        by_cols_dtypes : list-like, optional
            Data types of the internal 'by' columns. Required to do special casing
            in case of categorical 'by'. If not specified, assume that there is no
            categorical data in 'by'.
        by_length : int, optional
            Amount of keys to group on (including frame columns and external objects like list, Series, etc.)
            If not specified, consider `by_length` to be equal ``len(internal_by_cols)``.
        selection : label or list of labels, optional
            Set of columns that were explicitly selected for aggregation (for example
            via dict-aggregation). If not specified assuming that aggregation was
            applied to all of the available columns.
        partition_idx : int, default: 0
            Positional index of the current partition.
        drop : bool, default: True
            Indicates whether or not any of the `by` data came from the same frame.
        method : str, optional
            Name of the groupby function. This is a hint to be able to do special casing.
            Note: this parameter is a legacy from the ``groupby_size`` implementation,
            it's a hacky one and probably will be removed in the future: https://github.com/modin-project/modin/issues/3739.

        Returns
        -------
        reset_index : bool
            Indicates whether to reset index to the default one (0, 1, 2 ... n) at this partition.
        drop_index : bool
            If `reset_index` is True, indicates whether to drop all index levels (True) or insert them into the
            resulting columns (False).
        lvls_to_drop : list of ints
            Contains numeric indices of the levels of the result index to drop as intersected.
        cols_to_drop : list of labels
            Contains labels of the columns to drop from the result as intersected.

        Examples
        --------
        >>> groupby_result = compute_groupby_without_processing_as_index_parameter()
        >>> if not as_index:
        >>>     reset_index, drop, lvls_to_drop, cols_to_drop = handle_as_index(**extract_required_params(groupby_result))
        >>>     if len(lvls_to_drop) > 0:
        >>>         groupby_result.index = groupby_result.index.droplevel(lvls_to_drop)
        >>>     if len(cols_to_drop) > 0:
        >>>         groupby_result = groupby_result.drop(columns=cols_to_drop)
        >>>     if reset_index:
        >>>         groupby_result_with_processed_as_index_parameter = groupby_result.reset_index(drop=drop)
        >>> else:
        >>>     groupby_result_with_processed_as_index_parameter = groupby_result
        """
    if by_length is None:
        by_length = len(internal_by_cols)
    reset_index = method != 'transform' and (by_length > 0 or selection is not None)
    if method == 'size':
        return (reset_index, False, [], [])
    if by_cols_dtypes is not None:
        keep_index_levels = by_length > 1 and selection is None and any((isinstance(x, pandas.CategoricalDtype) for x in by_cols_dtypes))
    else:
        keep_index_levels = False
    if not keep_index_levels and partition_idx != 0 or (not drop and method != 'size'):
        return (reset_index, True, [], [])
    if not isinstance(internal_by_cols, pandas.Index):
        if not is_list_like(internal_by_cols):
            internal_by_cols = [internal_by_cols]
        internal_by_cols = pandas.Index(internal_by_cols)
    internal_by_cols = internal_by_cols[~internal_by_cols.str.startswith(MODIN_UNNAMED_SERIES_LABEL, na=False)] if hasattr(internal_by_cols, 'str') else internal_by_cols
    if selection is not None and (not isinstance(selection, pandas.Index)):
        selection = pandas.Index(selection)
    lvls_to_drop = []
    cols_to_drop = []
    if not keep_index_levels:
        if selection is None:
            cols_to_insert = frozenset(internal_by_cols) - frozenset(result_cols)
        else:
            cols_to_insert = frozenset((col for col in internal_by_cols if col not in selection))
    else:
        cols_to_insert = internal_by_cols
        cols_to_drop = frozenset(internal_by_cols) & frozenset(result_cols)
    if partition_idx == 0:
        lvls_to_drop = [i for i, name in enumerate(result_index_names) if name not in cols_to_insert]
    else:
        lvls_to_drop = result_index_names
    drop = False
    if len(lvls_to_drop) == len(result_index_names):
        drop = True
        lvls_to_drop = []
    return (reset_index, drop, lvls_to_drop, cols_to_drop)