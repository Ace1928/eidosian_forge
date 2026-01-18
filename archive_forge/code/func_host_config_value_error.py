from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
def host_config_value_error(param, param_value):
    error_msg = 'Invalid value for {0} param: {1}'
    return ValueError(error_msg.format(param, param_value))