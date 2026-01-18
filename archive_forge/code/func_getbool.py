from __future__ import annotations
import copy
import json
from importlib import import_module
from pprint import pformat
from types import ModuleType
from typing import (
from scrapy.settings import default_settings
def getbool(self, name: _SettingsKeyT, default: bool=False) -> bool:
    """
        Get a setting value as a boolean.

        ``1``, ``'1'``, `True`` and ``'True'`` return ``True``,
        while ``0``, ``'0'``, ``False``, ``'False'`` and ``None`` return ``False``.

        For example, settings populated through environment variables set to
        ``'0'`` will return ``False`` when using this method.

        :param name: the setting name
        :type name: str

        :param default: the value to return if no setting is found
        :type default: object
        """
    got = self.get(name, default)
    try:
        return bool(int(got))
    except ValueError:
        if got in ('True', 'true'):
            return True
        if got in ('False', 'false'):
            return False
        raise ValueError("Supported values for boolean settings are 0/1, True/False, '0'/'1', 'True'/'False' and 'true'/'false'")