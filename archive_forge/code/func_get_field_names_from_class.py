import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
@classmethod
def get_field_names_from_class(cls, item_class: type) -> Optional[List[str]]:
    adapter_class = cls._get_adapter_class(item_class)
    return adapter_class.get_field_names_from_class(item_class)