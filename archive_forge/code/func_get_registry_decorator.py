import importlib
import os
import sys
from collections import namedtuple
from dataclasses import fields
from typing import Any, Callable, Dict, List
def get_registry_decorator(class_registry, name_registry, reference_class, default_config) -> Callable[[str, Any], Callable[[Any], Any]]:

    def register_item(name: str, config: Any=default_config):
        """Registers a subclass.

        This decorator allows xFormers to instantiate a given subclass
        from a configuration file, even if the class itself is not part of the
        xFormers library."""

        def register_cls(cls):
            if name in class_registry:
                raise ValueError('Cannot register duplicate item ({})'.format(name))
            if not issubclass(cls, reference_class):
                raise ValueError('Item ({}: {}) must extend the base class: {}'.format(name, cls.__name__, reference_class.__name__))
            if cls.__name__ in name_registry:
                raise ValueError('Cannot register item with duplicate class name ({})'.format(cls.__name__))
            class_registry[name] = Item(constructor=cls, config=config)
            name_registry.add(cls.__name__)
            return cls
        return register_cls
    return register_item