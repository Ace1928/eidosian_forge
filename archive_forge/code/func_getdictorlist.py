from __future__ import annotations
import copy
import json
from importlib import import_module
from pprint import pformat
from types import ModuleType
from typing import (
from scrapy.settings import default_settings
def getdictorlist(self, name: _SettingsKeyT, default: Union[Dict[Any, Any], List[Any], Tuple[Any], None]=None) -> Union[Dict[Any, Any], List[Any]]:
    """Get a setting value as either a :class:`dict` or a :class:`list`.

        If the setting is already a dict or a list, a copy of it will be
        returned.

        If it is a string it will be evaluated as JSON, or as a comma-separated
        list of strings as a fallback.

        For example, settings populated from the command line will return:

        -   ``{'key1': 'value1', 'key2': 'value2'}`` if set to
            ``'{"key1": "value1", "key2": "value2"}'``

        -   ``['one', 'two']`` if set to ``'["one", "two"]'`` or ``'one,two'``

        :param name: the setting name
        :type name: string

        :param default: the value to return if no setting is found
        :type default: any
        """
    value = self.get(name, default)
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value_loaded = json.loads(value)
            assert isinstance(value_loaded, (dict, list))
            return value_loaded
        except ValueError:
            return value.split(',')
    if isinstance(value, tuple):
        return list(value)
    assert isinstance(value, (dict, list))
    return copy.deepcopy(value)