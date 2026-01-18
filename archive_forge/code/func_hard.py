from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
@hard.setter
def hard(self, value):
    self['Hard'] = value