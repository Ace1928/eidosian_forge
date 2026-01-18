import functools
import inspect
import logging
def log_method_call(method):
    """Decorator helping to log method calls.

    :param method: Method to decorate to be logged.
    :type method: method definition
    """
    log = logging.getLogger(method.__module__)

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        args_start_pos = 0
        if args:
            first_arg = args[0]
            if _is_method(first_arg, method):
                cls = first_arg if isinstance(first_arg, type) else first_arg.__class__
                caller = _get_full_class_name(cls)
                args_start_pos = 1
            else:
                caller = 'static'
        else:
            caller = 'static'
        data = {'caller': caller, 'method_name': method.__name__, 'args': args[args_start_pos:], 'kwargs': kwargs}
        log.debug('%(caller)s method %(method_name)s called with arguments %(args)s %(kwargs)s', data)
        return method(*args, **kwargs)
    return wrapper