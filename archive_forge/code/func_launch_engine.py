import eventlet
import __original_module_threading as orig_threading
import threading  # noqa
import sys
from oslo_concurrency import processutils
from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from oslo_reports import guru_meditation_report as gmr
from oslo_service import service
from heat.common import config
from heat.common import messaging
from heat.common import profiler
from heat.engine import template
from heat.rpc import api as rpc_api
from heat import version
def launch_engine(setup_logging=True):
    if setup_logging:
        logging.register_options(CONF)
    CONF(project='heat', prog='heat-engine', version=version.version_info.version_string())
    if setup_logging:
        logging.setup(CONF, CONF.prog)
        logging.set_defaults()
    LOG = logging.getLogger(CONF.prog)
    messaging.setup()
    config.startup_sanity_check()
    mgr = None
    try:
        mgr = template._get_template_extension_manager()
    except template.TemplatePluginNotRegistered as ex:
        LOG.critical('%s', ex)
    if not mgr or not mgr.names():
        sys.exit('ERROR: No template format plugins registered')
    from heat.engine import service as engine
    profiler.setup(CONF.prog, CONF.host)
    gmr.TextGuruMeditation.setup_autorun(version)
    srv = engine.EngineService(CONF.host, rpc_api.ENGINE_TOPIC)
    workers = CONF.num_engine_workers
    if not workers:
        workers = max(4, processutils.get_worker_count())
    launcher = service.launch(CONF, srv, workers=workers, restart_method='mutate')
    return launcher