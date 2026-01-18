import contextlib
import os
import toml
from appdirs import user_config_dir
@staticmethod
def safe_set(dct, value, *keys):
    """Safely set the value of a key from a nested dictionary.

        If any key provided does not exist, a dictionary containing the
        remaining keys is dynamically created and set to the required value.

        Args:
            dct (dict): the dictionary to set the value of.
            value: the value to set. Can be any valid type.
            *keys: each additional argument corresponds to a nested key.
        """
    for key in keys[:-1]:
        dct = dct.setdefault(key, {})
    dct[keys[-1]] = value