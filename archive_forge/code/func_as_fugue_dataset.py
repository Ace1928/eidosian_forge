from typing import Any, Optional
from .._utils.registry import fugue_plugin
from .dataset import AnyDataset, Dataset
@fugue_plugin
def as_fugue_dataset(data: AnyDataset, **kwargs: Any) -> Dataset:
    """Wrap the input as a :class:`~.Dataset`

    :param data: the dataset to be wrapped
    """
    if isinstance(data, Dataset) and len(kwargs) == 0:
        return data
    raise NotImplementedError(f'no registered dataset conversion for {type(data)}')