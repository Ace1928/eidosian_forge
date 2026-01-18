from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from heat.common import config
from heat.common import messaging
from heat.common import profiler
from heat import version
def init_application():
    i18n.enable_lazy()
    CONF.reset()
    logging.register_options(CONF)
    CONF(project='heat', prog='heat-api-cfn', version=version.version_info.version_string())
    logging.setup(CONF, CONF.prog)
    logging.set_defaults()
    LOG = logging.getLogger(CONF.prog)
    config.set_config_defaults()
    messaging.setup()
    port = CONF.heat_api_cfn.bind_port
    host = CONF.heat_api_cfn.bind_host
    LOG.info('Starting Heat API on %(host)s:%(port)s', {'host': host, 'port': port})
    profiler.setup(CONF.prog, host)
    return config.load_paste_app()