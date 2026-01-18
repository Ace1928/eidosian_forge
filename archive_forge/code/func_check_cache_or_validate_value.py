import collections
import json
import numbers
import re
from oslo_cache import core
from oslo_config import cfg
from oslo_log import log
from oslo_utils import reflection
from oslo_utils import strutils
from heat.common import cache
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resources
@MEMOIZE
def check_cache_or_validate_value(cache_value_prefix, value_to_validate):
    """Check if validation result stored in cache or validate value.

            The function checks that value was validated and validation
            result stored in cache. If not then it executes validation and
            stores the result of validation in cache.
            If caching is disabled it requests for validation each time.

            :param cache_value_prefix: cache prefix that used to distinguish
                                       value in heat cache. So the cache key
                                       would be the following:
                                       cache_value_prefix + value_to_validate.
            :param value_to_validate: value that need to be validated
            :return: True if value is valid otherwise False
            """
    try:
        self.validate_with_client(context.clients, value_to_validate)
    except self.expected_exceptions as e:
        self._error_message = str(e)
        return False
    else:
        return True