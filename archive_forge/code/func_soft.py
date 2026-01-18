from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
@soft.setter
def soft(self, value):
    self['Soft'] = value