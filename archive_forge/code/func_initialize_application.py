import collections
import os
from oslo_log import log
import stevedore
from keystone.common import profiler
import keystone.conf
import keystone.server
from keystone.server.flask import application
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.server.flask.request_processing.middleware import url_normalize
def initialize_application(name, post_log_configured_function=lambda: None, config_files=None):
    possible_topdir = os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))
    dev_conf = os.path.join(possible_topdir, 'etc', 'keystone.conf')
    if not config_files:
        config_files = None
        if os.path.exists(dev_conf):
            config_files = [dev_conf]
    keystone.server.configure(config_files=config_files)
    if CONF.debug:
        CONF.log_opt_values(log.getLogger(CONF.prog), log.DEBUG)
    post_log_configured_function()

    def loadapp():
        app = application.application_factory(name)
        return app
    _unused, app = keystone.server.setup_backends(startup_application_fn=loadapp)
    profiler.setup(name)
    return setup_app_middleware(app)