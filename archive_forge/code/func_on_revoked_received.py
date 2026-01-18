from celery import bootsteps
from celery.utils.log import get_logger
from .events import Events
def on_revoked_received(self, c, revoked):
    if revoked:
        c.controller.state.revoked.update(revoked)