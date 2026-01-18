from modin.config import Engine, IsExperimental, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.utils import _inherit_docstrings, get_current_execution
@classmethod
def set_failing_name(cls, factory_name):
    """
        Fill in `.io_cls` class attribute with ``StubIoEngine`` engine.

        Parameters
        ----------
        factory_name : str
            Name to pass to the ``StubIoEngine`` constructor.
        """
    cls.io_cls = StubIoEngine(factory_name)
    return cls