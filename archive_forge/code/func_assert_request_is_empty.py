from functools import partial
import numpy as np
from sklearn.base import (
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.multiclass import _check_partial_fit_first_call
def assert_request_is_empty(metadata_request, exclude=None):
    """Check if a metadata request dict is empty.

    One can exclude a method or a list of methods from the check using the
    ``exclude`` parameter. If metadata_request is a MetadataRouter, then
    ``exclude`` can be of the form ``{"object" : [method, ...]}``.
    """
    if isinstance(metadata_request, MetadataRouter):
        for name, route_mapping in metadata_request:
            if exclude is not None and name in exclude:
                _exclude = exclude[name]
            else:
                _exclude = None
            assert_request_is_empty(route_mapping.router, exclude=_exclude)
        return
    exclude = [] if exclude is None else exclude
    for method in SIMPLE_METHODS:
        if method in exclude:
            continue
        mmr = getattr(metadata_request, method)
        props = [prop for prop, alias in mmr.requests.items() if isinstance(alias, str) or alias is not None]
        assert not props