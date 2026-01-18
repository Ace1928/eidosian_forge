from __future__ import annotations
import copy
import json
from importlib import import_module
from pprint import pformat
from types import ModuleType
from typing import (
from scrapy.settings import default_settings
def setmodule(self, module: Union[ModuleType, str], priority: Union[int, str]='project') -> None:
    """
        Store settings from a module with a given priority.

        This is a helper function that calls
        :meth:`~scrapy.settings.BaseSettings.set` for every globally declared
        uppercase variable of ``module`` with the provided ``priority``.

        :param module: the module or the path of the module
        :type module: types.ModuleType or str

        :param priority: the priority of the settings. Should be a key of
            :attr:`~scrapy.settings.SETTINGS_PRIORITIES` or an integer
        :type priority: str or int
        """
    self._assert_mutability()
    if isinstance(module, str):
        module = import_module(module)
    for key in dir(module):
        if key.isupper():
            self.set(key, getattr(module, key), priority)