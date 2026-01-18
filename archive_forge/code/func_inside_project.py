import os
import warnings
from importlib import import_module
from pathlib import Path
from scrapy.exceptions import NotConfigured
from scrapy.settings import Settings
from scrapy.utils.conf import closest_scrapy_cfg, get_config, init_env
def inside_project() -> bool:
    scrapy_module = os.environ.get(ENVVAR)
    if scrapy_module:
        try:
            import_module(scrapy_module)
        except ImportError as exc:
            warnings.warn(f'Cannot import scrapy settings module {scrapy_module}: {exc}')
        else:
            return True
    return bool(closest_scrapy_cfg())