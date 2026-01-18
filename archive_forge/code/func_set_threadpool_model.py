import futurist
from oslo_log import log as logging
from glance.i18n import _LE
def set_threadpool_model(thread_type):
    """Set the system-wide threadpool model.

    This sets the type of ThreadPoolModel to use globally in the process.
    It should be called very early in init, and only once.

    :param thread_type: A string indicating the threading type in use,
                        either "eventlet" or "native"
    :raises: RuntimeError if the model is already set or some thread_type
             other than one of the supported ones is provided.
    """
    global _THREADPOOL_MODEL
    if thread_type == 'native':
        model = NativeThreadPoolModel
    elif thread_type == 'eventlet':
        model = EventletThreadPoolModel
    else:
        raise RuntimeError('Invalid thread type %r (must be "native" or "eventlet")' % thread_type)
    if _THREADPOOL_MODEL is model:
        return
    if _THREADPOOL_MODEL is not None:
        raise RuntimeError('Thread model is already set')
    LOG.info('Threadpool model set to %r', model.__name__)
    _THREADPOOL_MODEL = model