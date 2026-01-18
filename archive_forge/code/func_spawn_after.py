import logging
import os
from os_ken.lib import ip
def spawn_after(seconds, *args, **kwargs):
    raise_error = kwargs.pop('raise_error', False)

    def _launch(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TaskExit:
            pass
        except BaseException as e:
            if raise_error:
                raise e
            LOG.error('hub: uncaught exception: %s', traceback.format_exc())
    return eventlet.spawn_after(seconds, _launch, *args, **kwargs)