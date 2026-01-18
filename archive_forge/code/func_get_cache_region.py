from oslo_cache import core
from oslo_config import cfg
from heat.common.i18n import _
def get_cache_region():
    global _REGION
    if not _REGION:
        _REGION = core.configure_cache_region(conf=register_cache_configurations(cfg.CONF), region=core.create_region())
    return _REGION