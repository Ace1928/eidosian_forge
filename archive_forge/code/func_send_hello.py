from celery import bootsteps
from celery.utils.log import get_logger
from .events import Events
def send_hello(self, c):
    inspect = c.app.control.inspect(timeout=1.0, connection=c.connection)
    our_revoked = c.controller.state.revoked
    replies = inspect.hello(c.hostname, our_revoked._data) or {}
    replies.pop(c.hostname, None)
    return replies