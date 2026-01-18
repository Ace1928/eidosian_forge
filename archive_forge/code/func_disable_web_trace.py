from oslo_config import cfg
from osprofiler import web
def disable_web_trace(conf=None):
    if conf is None:
        conf = cfg.CONF
    if conf.profiler.enabled:
        web.disable()