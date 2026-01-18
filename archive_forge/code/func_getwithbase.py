from __future__ import annotations
import copy
import json
from importlib import import_module
from pprint import pformat
from types import ModuleType
from typing import (
from scrapy.settings import default_settings
def getwithbase(self, name: _SettingsKeyT) -> 'BaseSettings':
    """Get a composition of a dictionary-like setting and its `_BASE`
        counterpart.

        :param name: name of the dictionary-like setting
        :type name: str
        """
    if not isinstance(name, str):
        raise ValueError(f'Base setting key must be a string, got {name}')
    compbs = BaseSettings()
    compbs.update(self[name + '_BASE'])
    compbs.update(self[name])
    return compbs