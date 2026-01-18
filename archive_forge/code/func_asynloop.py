import errno
import socket
from celery import bootsteps
from celery.exceptions import WorkerLostError
from celery.utils.log import get_logger
from . import state
def asynloop(obj, connection, consumer, blueprint, hub, qos, heartbeat, clock, hbrate=2.0):
    """Non-blocking event loop."""
    RUN = bootsteps.RUN
    update_qos = qos.update
    errors = connection.connection_errors
    on_task_received = obj.create_task_handler()
    heartbeat_error = _enable_amqheartbeats(hub.timer, connection, rate=hbrate)
    consumer.on_message = on_task_received
    obj.controller.register_with_event_loop(hub)
    obj.register_with_event_loop(hub)
    consumer.consume()
    obj.on_ready()
    if not obj.restart_count and (not obj.pool.did_start_ok()):
        raise WorkerLostError('Could not start worker processes')
    if connection.transport.driver_type == 'amqp':
        hub.call_soon(_quick_drain, connection)
    hub.propagate_errors = errors
    loop = hub.create_loop()
    try:
        while blueprint.state == RUN and obj.connection:
            state.maybe_shutdown()
            if heartbeat_error[0] is not None:
                raise heartbeat_error[0]
            if qos.prev != qos.value:
                update_qos()
            try:
                next(loop)
            except StopIteration:
                loop = hub.create_loop()
    finally:
        try:
            hub.reset()
        except Exception as exc:
            logger.exception('Error cleaning up after event loop: %r', exc)