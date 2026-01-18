from typing import List, Optional, Tuple, Dict, Iterable, overload, Union
from altair import (
from altair.utils._vegafusion_data import get_inline_tables, import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.schemapi import Undefined
def get_from_facet_mapping(scoped_dataset: Tuple[str, Scope], facet_mapping: FacetMapping) -> Tuple[str, Scope]:
    """Apply facet mapping to a scoped dataset

    Parameters
    ----------
    scoped_dataset : (str, tuple of int)
        A dataset name and scope tuple
    facet_mapping : dict from (str, tuple of int) to (str, tuple of int)
        The facet mapping produced by get_facet_mapping

    Returns
    -------
    (str, tuple of int)
        Dataset name and scope tuple that has been mapped as many times as possible

    Examples
    --------
    Facet mapping as produced by get_facet_mapping
    >>> facet_mapping = {("facet1", (0,)): ("data1", ()), ("facet2", (0, 1)): ("facet1", (0,))}
    >>> get_from_facet_mapping(("facet2", (0, 1)), facet_mapping)
    ('data1', ())
    """
    while scoped_dataset in facet_mapping:
        scoped_dataset = facet_mapping[scoped_dataset]
    return scoped_dataset