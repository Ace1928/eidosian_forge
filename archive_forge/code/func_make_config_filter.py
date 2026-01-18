import re
import threading
def make_config_filter(app, global_conf, **local_conf):
    conf = global_conf.copy()
    conf.update(local_conf)
    return ConfigMiddleware(app, conf)