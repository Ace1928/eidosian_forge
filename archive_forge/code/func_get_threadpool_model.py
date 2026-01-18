import futurist
from oslo_log import log as logging
from glance.i18n import _LE
def get_threadpool_model():
    """Returns the system-wide threadpool model class.

    This must be called after set_threadpool_model() whenever
    some code needs to know what the threadpool implementation is.

    This may only be called after set_threadpool_model() has been
    called to set the desired threading mode. If it is called before
    the model is set, it will raise AssertionError. This would likely
    be the case if this got run in a test before the model was
    initialized, or if glance modules that use threading were imported
    and run from some other code without setting the model first.

    :raises: AssertionError if the model has not yet been set.
    """
    global _THREADPOOL_MODEL
    assert _THREADPOOL_MODEL
    return _THREADPOOL_MODEL