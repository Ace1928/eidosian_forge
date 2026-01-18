import os
import flask
from . import exceptions
from ._utils import AttributeDict
def get_combined_config(name, val, default=None):
    """Consolidate the config with priority from high to low provided init
    value > OS environ > default."""
    if val is not None:
        return val
    env = load_dash_env_vars().get(f'DASH_{name.upper()}')
    if env is None:
        return default
    return env.lower() == 'true' if env.lower() in {'true', 'false'} else env