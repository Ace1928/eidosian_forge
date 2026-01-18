import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
def get_field_meta(self, field_name: str) -> MappingProxyType:
    """Return metadata for the given field name."""
    return self.adapter.get_field_meta(field_name)