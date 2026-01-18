from typing import List, Optional, Tuple, Dict, Iterable, overload, Union
from altair import (
from altair.utils._vegafusion_data import get_inline_tables, import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.schemapi import Undefined
def get_facet_mapping(group: dict, scope: Scope=()) -> FacetMapping:
    """Create mapping from facet definitions to source datasets

    Parameters
    ----------
    group : dict
        Top-level Vega spec or nested group mark
    scope : tuple of int
        Scope of the group dictionary within a top-level Vega spec

    Returns
    -------
    dict
        Dictionary from (facet_name, facet_scope) to (dataset_name, dataset_scope)

    Examples
    --------
    >>> spec = {
    ...     "data": [
    ...         {"name": "data1"}
    ...     ],
    ...     "marks": [
    ...         {
    ...             "type": "group",
    ...             "from": {
    ...                 "facet": {
    ...                     "name": "facet1",
    ...                     "data": "data1",
    ...                     "groupby": ["colA"]
    ...                 }
    ...             }
    ...         }
    ...     ]
    ... }
    >>> get_facet_mapping(spec)
    {('facet1', (0,)): ('data1', ())}
    """
    facet_mapping = {}
    group_index = 0
    mark_group = get_group_mark_for_scope(group, scope) or {}
    for mark in mark_group.get('marks', []):
        if mark.get('type', None) == 'group':
            group_scope = scope + (group_index,)
            facet = mark.get('from', {}).get('facet', None)
            if facet is not None:
                facet_name = facet.get('name', None)
                facet_data = facet.get('data', None)
                if facet_name is not None and facet_data is not None:
                    definition_scope = get_definition_scope_for_data_reference(group, facet_data, scope)
                    if definition_scope is not None:
                        facet_mapping[facet_name, group_scope] = (facet_data, definition_scope)
            child_mapping = get_facet_mapping(group, scope=group_scope)
            facet_mapping.update(child_mapping)
            group_index += 1
    return facet_mapping