import os
import sys
import locale
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import MutableMapping, Mapping, Any, Dict
from xdg import BaseDirectory
from .autocomplete import AutocompleteModes
def get_key_no_doublebind(command: str) -> str:
    default_commands_to_keys = self.defaults['keyboard']
    requested_key = config.get('keyboard', command)
    try:
        default_command = default_keys_to_commands[requested_key]
        if default_commands_to_keys[default_command] == config.get('keyboard', default_command):
            setattr(self, f'{default_command}_key', '')
    except KeyError:
        pass
    return requested_key