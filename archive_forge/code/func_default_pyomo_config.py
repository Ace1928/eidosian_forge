import os
import json
import pyomo.common.envvar as envvar
from pyomo.common.config import ConfigBase, ConfigBlock, ConfigValue, ADVANCED_OPTION
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import logging
def default_pyomo_config():
    config = ConfigBlock('Pyomo configuration file')
    config.declare('paranoia_level', ConfigValue(0, int, 'Pyomo paranoia and error checking level', 'Higher levels of paranoia enable additional error checking and\n            warning messages that may assist users in identifying likely\n            modeling problems.\n            Default=0', visibility=ADVANCED_OPTION))
    return config