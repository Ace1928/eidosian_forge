import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
def flatten_inference_data_to_dict(data, var_names=None, groups=None, dimensions=None, group_info=False, var_name_format=None, index_origin=None):
    """Transform data to dictionary.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_inference_data for details
    var_names : str or list of str, optional
        Variables to be processed, if None all variables are processed.
    groups : str or list of str, optional
        Select groups for CDS. Default groups are
        {"posterior_groups", "prior_groups", "posterior_groups_warmup"}
            - posterior_groups: posterior, posterior_predictive, sample_stats
            - prior_groups: prior, prior_predictive, sample_stats_prior
            - posterior_groups_warmup: warmup_posterior, warmup_posterior_predictive,
                                       warmup_sample_stats
    ignore_groups : str or list of str, optional
        Ignore specific groups from CDS.
    dimension : str, or list of str, optional
        Select dimensions along to slice the data. By default uses ("chain", "draw").
    group_info : bool
        Add group info for `var_name_format`
    var_name_format : str or tuple of tuple of string, optional
        Select column name format for non-scalar input.
        Predefined options are {"brackets", "underscore", "cds"}
            "brackets":
                - add_group_info == False: theta[0,0]
                - add_group_info == True: theta_posterior[0,0]
            "underscore":
                - add_group_info == False: theta_0_0
                - add_group_info == True: theta_posterior_0_0_
            "cds":
                - add_group_info == False: theta_ARVIZ_CDS_SELECTION_0_0
                - add_group_info == True: theta_ARVIZ_GROUP_posterior__ARVIZ_CDS_SELECTION_0_0
            tuple:
                Structure:
                    tuple: (dim_info, group_info)
                        dim_info: (str: `.join` separator,
                                   str: dim_separator_start,
                                   str: dim_separator_end)
                        group_info: (str: group separator start, str: group separator end)
                Example: ((",", "[", "]"), ("_", ""))
                    - add_group_info == False: theta[0,0]
                    - add_group_info == True: theta_posterior[0,0]
    index_origin : int, optional
        Start parameter indices from `index_origin`. Either 0 or 1.

    Returns
    -------
    dict
    """
    from .data import convert_to_inference_data
    data = convert_to_inference_data(data)
    if groups is None:
        groups = ['posterior', 'posterior_predictive', 'sample_stats']
    elif isinstance(groups, str):
        if groups.lower() == 'posterior_groups':
            groups = ['posterior', 'posterior_predictive', 'sample_stats']
        elif groups.lower() == 'prior_groups':
            groups = ['prior', 'prior_predictive', 'sample_stats_prior']
        elif groups.lower() == 'posterior_groups_warmup':
            groups = ['warmup_posterior', 'warmup_posterior_predictive', 'warmup_sample_stats']
        else:
            raise TypeError('Valid predefined groups are {posterior_groups, prior_groups, posterior_groups_warmup}')
    if dimensions is None:
        dimensions = ('chain', 'draw')
    elif isinstance(dimensions, str):
        dimensions = (dimensions,)
    if var_name_format is None:
        var_name_format = 'brackets'
    if isinstance(var_name_format, str):
        var_name_format = var_name_format.lower()
    if var_name_format == 'brackets':
        dim_join_separator, dim_separator_start, dim_separator_end = (',', '[', ']')
        group_separator_start, group_separator_end = ('_', '')
    elif var_name_format == 'underscore':
        dim_join_separator, dim_separator_start, dim_separator_end = ('_', '_', '')
        group_separator_start, group_separator_end = ('_', '')
    elif var_name_format == 'cds':
        dim_join_separator, dim_separator_start, dim_separator_end = ('_', '_ARVIZ_CDS_SELECTION_', '')
        group_separator_start, group_separator_end = ('_ARVIZ_GROUP_', '')
    elif isinstance(var_name_format, str):
        msg = 'Invalid predefined format. Select one {"brackets", "underscore", "cds"}'
        raise TypeError(msg)
    else:
        (dim_join_separator, dim_separator_start, dim_separator_end), (group_separator_start, group_separator_end) = var_name_format
    if index_origin is None:
        index_origin = rcParams['data.index_origin']
    data_dict = {}
    for group in groups:
        if hasattr(data, group):
            group_data = getattr(data, group).stack(stack_dimension=dimensions)
            for var_name, var in group_data.data_vars.items():
                var_values = var.values
                if var_names is not None and var_name not in var_names:
                    continue
                for dim_name in dimensions:
                    if dim_name not in data_dict:
                        data_dict[dim_name] = var.coords.get(dim_name).values
                if len(var.shape) == 1:
                    if group_info:
                        var_name_dim = '{var_name}{group_separator_start}{group}{group_separator_end}'.format(var_name=var_name, group_separator_start=group_separator_start, group=group, group_separator_end=group_separator_end)
                    else:
                        var_name_dim = f'{var_name}'
                    data_dict[var_name_dim] = var.values
                else:
                    for loc in np.ndindex(var.shape[:-1]):
                        if group_info:
                            var_name_dim = '{var_name}{group_separator_start}{group}{group_separator_end}{dim_separator_start}{dim_join}{dim_separator_end}'.format(var_name=var_name, group_separator_start=group_separator_start, group=group, group_separator_end=group_separator_end, dim_separator_start=dim_separator_start, dim_join=dim_join_separator.join((str(item + index_origin) for item in loc)), dim_separator_end=dim_separator_end)
                        else:
                            var_name_dim = '{var_name}{dim_separator_start}{dim_join}{dim_separator_end}'.format(var_name=var_name, dim_separator_start=dim_separator_start, dim_join=dim_join_separator.join((str(item + index_origin) for item in loc)), dim_separator_end=dim_separator_end)
                        data_dict[var_name_dim] = var_values[loc]
    return data_dict