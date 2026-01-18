import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
def interpret_config(self, config: 'ConfigParser') -> None:
    """Interpret a config, parse nested sections and parse the values
        as JSON. Mostly used internally and modifies the config in place.
        """
    self._validate_sections(config)
    get_depth = lambda item: len(item[0].split('.'))
    for section, values in sorted(config.items(), key=get_depth):
        if section == 'DEFAULT':
            continue
        parts = section.split('.')
        node = self
        for part in parts[:-1]:
            if part == '*':
                node = node.setdefault(part, {})
            elif part not in node:
                err_title = 'Error parsing config section. Perhaps a section name is wrong?'
                err = [{'loc': parts, 'msg': f"Section '{part}' is not defined"}]
                raise ConfigValidationError(config=self, errors=err, title=err_title)
            else:
                node = node[part]
        if not isinstance(node, dict):
            err = [{'loc': parts, 'msg': 'found conflicting values'}]
            err_cfg = f'{self}\n{ {part: dict(values)}}'
            raise ConfigValidationError(config=err_cfg, errors=err)
        node = node.setdefault(parts[-1], {})
        if not isinstance(node, dict):
            err = [{'loc': parts, 'msg': 'found conflicting values'}]
            err_cfg = f'{self}\n{ {part: dict(values)}}'
            raise ConfigValidationError(config=err_cfg, errors=err)
        try:
            keys_values = list(values.items())
        except InterpolationMissingOptionError as e:
            raise ConfigValidationError(desc=f'{e}') from None
        for key, value in keys_values:
            config_v = config.get(section, key)
            node[key] = self._interpret_value(config_v)
    self.replace_section_refs(self)