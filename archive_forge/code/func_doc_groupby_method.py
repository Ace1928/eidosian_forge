from functools import partial
from modin.utils import align_indents, append_to_docstring, format_string
def doc_groupby_method(result, refer_to, action=None):
    """
    Build decorator which adds docstring for the groupby reduce method.

    Parameters
    ----------
    result : str
        The result of reduce.
    refer_to : str
        Method name in ``modin.pandas.groupby`` module to refer to
        for more information about parameters and output format.
    action : str, optional
        What method does with groups.

    Returns
    -------
    callable
    """
    template = '\n    Group QueryCompiler data and {action} for every group.\n\n    Parameters\n    ----------\n    by : BaseQueryCompiler, column or index label, Grouper or list of such\n        Object that determine groups.\n    axis : {{0, 1}}\n        Axis to group and apply aggregation function along.\n        0 is for index, when 1 is for columns.\n    groupby_kwargs : dict\n        GroupBy parameters as expected by ``modin.pandas.DataFrame.groupby`` signature.\n    agg_args : list-like\n        Positional arguments to pass to the `agg_func`.\n    agg_kwargs : dict\n        Key arguments to pass to the `agg_func`.\n    drop : bool, default: False\n        If `by` is a QueryCompiler indicates whether or not by-data came\n        from the `self`.\n\n    Returns\n    -------\n    BaseQueryCompiler\n        QueryCompiler containing the result of groupby reduce built by the\n        following rules:\n\n        - Labels on the opposite of specified axis are preserved.\n        - If groupby_args["as_index"] is True then labels on the specified axis\n          are the group names, otherwise labels would be default: 0, 1 ... n.\n        - If groupby_args["as_index"] is False, then first N columns/rows of the frame\n          contain group names, where N is the columns/rows to group on.\n        - Each element of QueryCompiler is the {result} for the\n          corresponding group and column/row.\n\n    .. warning\n        `map_args` and `reduce_args` parameters are deprecated. They\'re leaked here from\n        ``PandasQueryCompiler.groupby_*``, pandas storage format implements groupby via TreeReduce\n        approach, but for other storage formats these parameters make no sense, and so they\'ll be removed in the future.\n    '
    if action is None:
        action = f'compute {result}'
    return doc_qc_method(template, result=result, action=action, refer_to=f'GroupBy.{refer_to}')