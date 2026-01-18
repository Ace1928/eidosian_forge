import os
import sys
import locale
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import MutableMapping, Mapping, Any, Dict
from xdg import BaseDirectory
from .autocomplete import AutocompleteModes
def load_theme(path: Path, colors: MutableMapping[str, str], default_colors: Mapping[str, str]) -> None:
    theme = ConfigParser()
    with open(path) as f:
        theme.read_file(f)
    for k, v in chain(theme.items('syntax'), theme.items('interface')):
        if theme.has_option('syntax', k):
            colors[k] = theme.get('syntax', k)
        else:
            colors[k] = theme.get('interface', k)
        if colors[k].lower() not in COLOR_LETTERS:
            raise UnknownColorCode(k, colors[k])
    for k, v in default_colors.items():
        if k not in colors:
            colors[k] = v