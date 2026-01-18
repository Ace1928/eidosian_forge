from __future__ import annotations
import json
from os import makedirs
from os.path import exists, expanduser
from pymatgen.analysis.chemenv.utils.scripts_utils import strategies_class_lookup
from pymatgen.core import SETTINGS
def package_options_description(self):
    """Describe package options."""
    out = 'Package options :\n'
    out += f' - Maximum distance factor : {self.package_options['default_max_distance_factor']:.4f}\n'
    out += f' - Default strategy is "{self.package_options['default_strategy']['strategy']}" :\n'
    strategy_class = strategies_class_lookup[self.package_options['default_strategy']['strategy']]
    out += f'{strategy_class.STRATEGY_DESCRIPTION}\n'
    out += '   with options :\n'
    for option in strategy_class.STRATEGY_OPTIONS:
        out += f'     - {option} : {self.package_options['default_strategy']['strategy_options'][option]}\n'
    return out