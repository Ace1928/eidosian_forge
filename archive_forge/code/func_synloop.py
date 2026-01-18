import errno
import socket
from celery import bootsteps
from celery.exceptions import WorkerLostError
from celery.utils.log import get_logger
from . import state
def synloop(obj, connection, consumer, blueprint, hub, qos, heartbeat, clock, hbrate=2.0, **kwargs):
    """Fallback blocking event loop for transports that doesn't support AIO."""
    RUN = bootsteps.RUN
    on_task_received = obj.create_task_handler()
    perform_pending_operations = obj.perform_pending_operations
    heartbeat_error = [None]
    if getattr(obj.pool, 'is_green', False):
        heartbeat_error = _enable_amqheartbeats(obj.timer, connection, rate=hbrate)
    consumer.on_message = on_task_received
    consumer.consume()
    obj.on_ready()
    while blueprint.state == RUN and obj.connection:
        state.maybe_shutdown()
        if heartbeat_error[0] is not None:
            raise heartbeat_error[0]
        if qos.prev != qos.value:
            qos.update()
        try:
            perform_pending_operations()
            connection.drain_events(timeout=2.0)
        except socket.timeout:
            pass
        except OSError:
            if blueprint.state == RUN:
                raise