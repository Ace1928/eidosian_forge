from __future__ import annotations
import copy
import json
from importlib import import_module
from pprint import pformat
from types import ModuleType
from typing import (
from scrapy.settings import default_settings
def getpriority(self, name: _SettingsKeyT) -> Optional[int]:
    """
        Return the current numerical priority value of a setting, or ``None`` if
        the given ``name`` does not exist.

        :param name: the setting name
        :type name: str
        """
    if name not in self:
        return None
    return self.attributes[name].priority