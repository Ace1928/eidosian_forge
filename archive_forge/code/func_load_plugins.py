import logging
import os
import json
from abc import ABC
from typing import List, Dict, Optional, Any, Type
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.uri_cache import URICache
from ray._private.runtime_env.constants import (
from ray.util.annotations import DeveloperAPI
from ray._private.utils import import_attr
def load_plugins(self, plugin_configs: List[Dict]) -> None:
    """Load runtime env plugins and create URI caches for them."""
    for plugin_config in plugin_configs:
        if not isinstance(plugin_config, dict) or RAY_RUNTIME_ENV_CLASS_FIELD_NAME not in plugin_config:
            raise RuntimeError(f'Invalid runtime env plugin config {plugin_config}, it should be a object which contains the {RAY_RUNTIME_ENV_CLASS_FIELD_NAME} field.')
        plugin_class = import_attr(plugin_config[RAY_RUNTIME_ENV_CLASS_FIELD_NAME])
        self.validate_plugin_class(plugin_class)
        if RAY_RUNTIME_ENV_PRIORITY_FIELD_NAME in plugin_config:
            priority = plugin_config[RAY_RUNTIME_ENV_PRIORITY_FIELD_NAME]
        else:
            priority = plugin_class.priority
        self.validate_priority(priority)
        class_instance = plugin_class()
        self.plugins[plugin_class.name] = PluginSetupContext(plugin_class.name, class_instance, priority, self.create_uri_cache_for_plugin(class_instance))