from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
def host_config_version_error(param, version, less_than=True):
    operator = '<' if less_than else '>'
    error_msg = '{0} param is not supported in API versions {1} {2}'
    return errors.InvalidVersion(error_msg.format(param, operator, version))