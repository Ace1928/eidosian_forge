import numbers
import os
import sys
import warnings
from configparser import ConfigParser
from operator import itemgetter
from pathlib import Path
from typing import (
from scrapy.exceptions import ScrapyDeprecationWarning, UsageError
from scrapy.settings import BaseSettings
from scrapy.utils.deprecate import update_classpath
from scrapy.utils.python import without_none_values
def get_sources(use_closest: bool=True) -> List[str]:
    xdg_config_home = os.environ.get('XDG_CONFIG_HOME') or Path('~/.config').expanduser()
    sources = ['/etc/scrapy.cfg', 'c:\\scrapy\\scrapy.cfg', str(Path(xdg_config_home) / 'scrapy.cfg'), str(Path('~/.scrapy.cfg').expanduser())]
    if use_closest:
        sources.append(closest_scrapy_cfg())
    return sources