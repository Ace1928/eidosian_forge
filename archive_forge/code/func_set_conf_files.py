from __future__ import annotations
import configparser
import json
import os
import warnings
from typing import Any
def set_conf_files(cdir, conf_dict):
    """Set config values from files

    Scans for INI and JSON files in the given dictionary, and uses their
    contents to set the config. In case of repeated values, later values
    win.

    In the case of INI files, all values are strings, and these will not
    be converted.

    Parameters
    ----------
    cdir : str
        Directory to search
    conf_dict : dict(str, dict)
        This dict will be mutated
    """
    if not os.path.isdir(cdir):
        return
    allfiles = sorted(os.listdir(cdir))
    for fn in allfiles:
        if fn.endswith('.ini'):
            ini = configparser.ConfigParser()
            ini.read(os.path.join(cdir, fn))
            for key in ini:
                if key == 'DEFAULT':
                    continue
                conf_dict.setdefault(key, {}).update(dict(ini[key]))
        if fn.endswith('.json'):
            with open(os.path.join(cdir, fn)) as f:
                js = json.load(f)
            for key in js:
                conf_dict.setdefault(key, {}).update(dict(js[key]))