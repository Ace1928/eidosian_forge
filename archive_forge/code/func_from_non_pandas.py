from modin.config import Engine, IsExperimental, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.utils import _inherit_docstrings, get_current_execution
@classmethod
@_inherit_docstrings(factories.BaseFactory._from_non_pandas)
def from_non_pandas(cls, *args, **kwargs):
    return cls.get_factory()._from_non_pandas(*args, **kwargs)