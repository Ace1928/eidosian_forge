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
def load_apps(self, app_lists):
    app_lists = [app for app in itertools.chain.from_iterable((app.split(',') for app in app_lists))]
    while len(app_lists) > 0:
        app_cls_name = app_lists.pop(0)
        context_modules = [x.__module__ for x in self.contexts_cls.values()]
        if app_cls_name in context_modules:
            continue
        LOG.info('loading app %s', app_cls_name)
        cls = self.load_app(app_cls_name)
        if cls is None:
            continue
        self.applications_cls[app_cls_name] = cls
        services = []
        for key, context_cls in cls.context_iteritems():
            v = self.contexts_cls.setdefault(key, context_cls)
            assert v == context_cls
            context_modules.append(context_cls.__module__)
            if issubclass(context_cls, OSKenApp):
                services.extend(get_dependent_services(context_cls))
        for i in get_dependent_services(cls):
            if i not in context_modules:
                services.append(i)
        if services:
            app_lists.extend([s for s in set(services) if s not in app_lists])