from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
def set_config_value(self, key, value):
    """ Set a the value for ``key`` to ``value`` inside the ``config``
            dict.
        """
    self.config[key] = value