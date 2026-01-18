import json
import logging
import os
from typing import Any, Dict, Optional
import yaml
import wandb
from wandb.errors import Error
from wandb.util import load_yaml
from . import filesystem
def dict_from_config_file(filename: str, must_exist: bool=False) -> Optional[Dict[str, Any]]:
    if not os.path.exists(filename):
        if must_exist:
            raise ConfigError("config file %s doesn't exist" % filename)
        logger.debug('no default config file found in %s' % filename)
        return None
    try:
        conf_file = open(filename)
    except OSError:
        raise ConfigError("Couldn't read config file: %s" % filename)
    try:
        loaded = load_yaml(conf_file)
    except yaml.parser.ParserError:
        raise ConfigError('Invalid YAML in config yaml')
    if loaded is None:
        wandb.termwarn('Found an empty default config file (config-defaults.yaml). Proceeding with no defaults.')
        return None
    config_version = loaded.pop('wandb_version', None)
    if config_version is not None and config_version != 1:
        raise ConfigError('Unknown config version')
    data = dict()
    for k, v in loaded.items():
        data[k] = v['value']
    return data