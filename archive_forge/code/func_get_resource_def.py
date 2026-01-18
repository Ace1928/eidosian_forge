import collections
import copy
import functools
import itertools
import math
from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import timeutils
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import function
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import support
from heat.scaling import rolling_update
from heat.scaling import template as scl_template
def get_resource_def(self, include_all=False):
    """Returns the resource definition portion of the group.

        :param include_all: if False, only properties for the resource
               definition that are not empty will be included
        :type include_all: bool
        :return: resource definition for the group
        :rtype: dict
        """

    def ignore_param_resolve(snippet, nullable=False):
        if isinstance(snippet, function.Function):
            try:
                result = snippet.result()
            except exception.UserParameterMissing:
                return None
            if not (nullable or function._non_null_value(result)):
                result = None
            return result
        if isinstance(snippet, collections.abc.Mapping):
            return dict(filter(function._non_null_item, ((k, ignore_param_resolve(v, nullable=True)) for k, v in snippet.items())))
        elif not isinstance(snippet, str) and isinstance(snippet, collections.abc.Iterable):
            return list(filter(function._non_null_value, (ignore_param_resolve(v, nullable=True) for v in snippet)))
        return snippet
    self.properties.resolve = ignore_param_resolve
    res_def = self.properties[self.RESOURCE_DEF]
    if not include_all:
        return self._clean_props(res_def)
    return res_def