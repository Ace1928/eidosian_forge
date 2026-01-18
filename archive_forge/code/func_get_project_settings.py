import os
import warnings
from importlib import import_module
from pathlib import Path
from scrapy.exceptions import NotConfigured
from scrapy.settings import Settings
from scrapy.utils.conf import closest_scrapy_cfg, get_config, init_env
def get_project_settings() -> Settings:
    if ENVVAR not in os.environ:
        project = os.environ.get('SCRAPY_PROJECT', 'default')
        init_env(project)
    settings = Settings()
    settings_module_path = os.environ.get(ENVVAR)
    if settings_module_path:
        settings.setmodule(settings_module_path, priority='project')
    valid_envvars = {'CHECK', 'PROJECT', 'PYTHON_SHELL', 'SETTINGS_MODULE'}
    scrapy_envvars = {k[7:]: v for k, v in os.environ.items() if k.startswith('SCRAPY_') and k.replace('SCRAPY_', '') in valid_envvars}
    settings.setdict(scrapy_envvars, priority='project')
    return settings