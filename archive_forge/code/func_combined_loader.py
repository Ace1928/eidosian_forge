from typing import Any, Iterator, List, Optional
from typing_extensions import override
from lightning_fabric.utilities.data import sized_len
from pytorch_lightning.utilities.combined_loader import _ITERATOR_RETURN, CombinedLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
@property
def combined_loader(self) -> CombinedLoader:
    if self._combined_loader is None:
        raise MisconfigurationException(f'`{self.__class__.__name__}` should have been `setup` with a `CombinedLoader`.')
    return self._combined_loader