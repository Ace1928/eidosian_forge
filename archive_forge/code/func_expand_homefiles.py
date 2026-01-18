import os
import re
import logging
import collections
import pyzor.account
def expand_homefiles(homefiles, category, homedir, config):
    """Set the full file path for these configuration files."""
    for filename in homefiles:
        filepath = config.get(category, filename)
        if not filepath:
            continue
        filepath = os.path.expanduser(filepath)
        if not os.path.isabs(filepath):
            filepath = os.path.join(homedir, filepath)
        config.set(category, filename, filepath)