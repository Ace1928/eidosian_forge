import inspect
import itertools
import logging
import sys
import os
import gc
from os_ken import cfg
from os_ken import utils
from os_ken.controller.handler import register_instance, get_dependent_services
from os_ken.controller.controller import Datapath
from os_ken.controller import event
from os_ken.controller.event import EventRequestBase, EventReplyBase
from os_ken.lib import hub
from os_ken.ofproto import ofproto_protocol
def require_app(app_name, api_style=False):
    """
    Request the application to be automatically loaded.

    If this is used for "api" style modules, which is imported by a client
    application, set api_style=True.

    If this is used for client application module, set api_style=False.
    """
    iterable = (inspect.getmodule(frame[0]) for frame in inspect.stack())
    modules = [module for module in iterable if module is not None]
    if api_style:
        m = modules[2]
    else:
        m = modules[1]
    m._REQUIRED_APP = getattr(m, '_REQUIRED_APP', [])
    m._REQUIRED_APP.append(app_name)
    LOG.debug('require_app: %s is required by %s', app_name, m.__name__)