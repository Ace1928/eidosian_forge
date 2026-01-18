from modin.config import Engine, IsExperimental, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.utils import _inherit_docstrings, get_current_execution
@classmethod
@_inherit_docstrings(factories.BaseFactory._read_fwf)
def read_fwf(cls, **kwargs):
    return cls.get_factory()._read_fwf(**kwargs)