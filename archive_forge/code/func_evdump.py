import sys
from datetime import datetime
from celery.app import app_or_default
from celery.utils.functional import LRUCache
from celery.utils.time import humanize_seconds
def evdump(app=None, out=sys.stdout):
    """Start event dump."""
    app = app_or_default(app)
    dumper = Dumper(out=out)
    dumper.say('-> evdump: starting capture...')
    conn = app.connection_for_read().clone()

    def _error_handler(exc, interval):
        dumper.say(CONNECTION_ERROR % (conn.as_uri(), exc, humanize_seconds(interval, 'in', ' ')))
    while 1:
        try:
            conn.ensure_connection(_error_handler)
            recv = app.events.Receiver(conn, handlers={'*': dumper.on_event})
            recv.capture()
        except (KeyboardInterrupt, SystemExit):
            return conn and conn.close()
        except conn.connection_errors + conn.channel_errors:
            dumper.say('-> Connection lost, attempting reconnect')