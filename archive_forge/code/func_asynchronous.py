import collections
import functools
import inspect
from oslo_log import log as logging
from oslo_messaging import rpc
def asynchronous(function):
    """Decorator for MessageProcessor methods to make them asynchronous.

    To use, simply call the method as usual. Instead of being executed
    immediately, it will be placed on the queue for the MessageProcessor and
    run on a future iteration of the event loop.
    """
    sig = inspect.signature(function)
    arg_names = [name for name, param in sig.parameters.items() if param.kind == param.POSITIONAL_OR_KEYWORD]
    MessageData = collections.namedtuple(function.__name__, arg_names[1:])

    @functools.wraps(function)
    def call_or_send(processor, *args, **kwargs):
        if len(args) == 1 and (not kwargs) and isinstance(args[0], MessageData):
            try:
                return function(processor, **args[0]._asdict())
            except rpc.dispatcher.ExpectedException as exc:
                LOG.error('[%s] Exception in "%s": %s', processor.name, function.__name__, exc.exc_info[1], exc_info=exc.exc_info)
                raise
            except Exception as exc:
                LOG.exception('[%s] Exception in "%s": %s', processor.name, function.__name__, exc)
                raise
        else:
            data = inspect.getcallargs(function, processor, *args, **kwargs)
            data.pop(arg_names[0])
            return processor.queue.send(function.__name__, MessageData(**data))
    call_or_send.MessageData = MessageData
    return call_or_send