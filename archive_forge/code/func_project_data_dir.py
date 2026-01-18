import os
import warnings
from importlib import import_module
from pathlib import Path
from scrapy.exceptions import NotConfigured
from scrapy.settings import Settings
from scrapy.utils.conf import closest_scrapy_cfg, get_config, init_env
def project_data_dir(project: str='default') -> str:
    """Return the current project data dir, creating it if it doesn't exist"""
    if not inside_project():
        raise NotConfigured('Not inside a project')
    cfg = get_config()
    if cfg.has_option(DATADIR_CFG_SECTION, project):
        d = Path(cfg.get(DATADIR_CFG_SECTION, project))
    else:
        scrapy_cfg = closest_scrapy_cfg()
        if not scrapy_cfg:
            raise NotConfigured('Unable to find scrapy.cfg file to infer project data dir')
        d = (Path(scrapy_cfg).parent / '.scrapy').resolve()
    if not d.exists():
        d.mkdir(parents=True)
    return str(d)