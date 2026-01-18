from celery import _state
from celery._state import app_or_default, disable_trace, enable_trace, pop_current_task, push_current_task
from celery.local import Proxy
from .base import Celery
from .utils import AppPickler
def shared_task(*args, **kwargs):
    """Create shared task (decorator).

    This can be used by library authors to create tasks that'll work
    for any app environment.

    Returns:
        ~celery.local.Proxy: A proxy that always takes the task from the
        current apps task registry.

    Example:

        >>> from celery import Celery, shared_task
        >>> @shared_task
        ... def add(x, y):
        ...     return x + y
        ...
        >>> app1 = Celery(broker='amqp://')
        >>> add.app is app1
        True
        >>> app2 = Celery(broker='redis://')
        >>> add.app is app2
        True
    """

    def create_shared_task(**options):

        def __inner(fun):
            name = options.get('name')
            _state.connect_on_app_finalize(lambda app: app._task_from_fun(fun, **options))
            for app in _state._get_active_apps():
                if app.finalized:
                    with app._finalize_mutex:
                        app._task_from_fun(fun, **options)

            def task_by_cons():
                app = _state.get_current_app()
                return app.tasks[name or app.gen_task_name(fun.__name__, fun.__module__)]
            return Proxy(task_by_cons)
        return __inner
    if len(args) == 1 and callable(args[0]):
        return create_shared_task(**kwargs)(args[0])
    return create_shared_task(*args, **kwargs)