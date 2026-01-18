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
def instantiate_apps(self, *args, **kwargs):
    for app_name, cls in self.applications_cls.items():
        self._instantiate(app_name, cls, *args, **kwargs)
    self._update_bricks()
    self.report_bricks()
    threads = []
    for app in self.applications.values():
        t = app.start()
        if t is not None:
            app.set_main_thread(t)
            threads.append(t)
    return threads