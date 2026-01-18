import copy
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from traceback import format_exc
from ..base import is_regressor
from ..utils import Bunch
from ..utils._param_validation import HasMethods, Hidden, StrOptions, validate_params
from ..utils._response import _get_response_values
from ..utils.metadata_routing import (
from ..utils.validation import _check_response_method
from . import (
from .cluster import (
def set_score_request(self, **kwargs):
    """Set requested parameters by the scorer.

        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Parameters
        ----------
        kwargs : dict
            Arguments should be of the form ``param_name=alias``, and `alias`
            can be one of ``{True, False, None, str}``.
        """
    if not _routing_enabled():
        raise RuntimeError('This method is only available when metadata routing is enabled. You can enable it using sklearn.set_config(enable_metadata_routing=True).')
    self._warn_overlap(message='You are setting metadata request for parameters which are already set as kwargs for this metric. These set values will be overridden by passed metadata if provided. Please pass them either as metadata or kwargs to `make_scorer`.', kwargs=kwargs)
    self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
    for param, alias in kwargs.items():
        self._metadata_request.score.add_request(param=param, alias=alias)
    return self