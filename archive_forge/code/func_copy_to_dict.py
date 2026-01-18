from __future__ import annotations
import copy
import json
from importlib import import_module
from pprint import pformat
from types import ModuleType
from typing import (
from scrapy.settings import default_settings
def copy_to_dict(self) -> Dict[_SettingsKeyT, Any]:
    """
        Make a copy of current settings and convert to a dict.

        This method returns a new dict populated with the same values
        and their priorities as the current settings.

        Modifications to the returned dict won't be reflected on the original
        settings.

        This method can be useful for example for printing settings
        in Scrapy shell.
        """
    settings = self.copy()
    return settings._to_dict()