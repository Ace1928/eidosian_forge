import os
import shutil
import importlib
from ase.calculators.calculator import names
def format_configs(configs):
    messages = []
    for name in names:
        config = configs.get(name)
        if config is None:
            state = 'no'
        else:
            type = config['type']
            if type == 'builtin':
                state = 'yes, builtin: module ase.calculators.{name}'
            elif type == 'python':
                state = 'yes, python: {module} ▶ {path}'
            elif type == 'which':
                state = 'yes, shell command: {command} ▶ {fullpath}'
            else:
                state = 'yes, environment: ${envvar} ▶ {command}'
            state = state.format(**config)
        messages.append('{:<10s} {}'.format(name, state))
    return messages