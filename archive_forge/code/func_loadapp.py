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
def loadapp():
    app = application.application_factory(name)
    return app