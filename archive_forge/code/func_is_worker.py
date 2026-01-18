import logging
from tornado import web
from . import BaseApiHandler
def is_worker(self, workername):
    return workername and workername in self.application.workers