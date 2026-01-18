from typing import (
from torch import Tensor, nn
from ..microbatch import Batch
from .namespace import Namespace
from .tracker import current_skip_tracker
def extend_skippable(module_cls: Type[SkippableModule]) -> Type[Skippable]:
    name = module_cls.__name__
    bases = (Skippable,)
    attrs = {'module_cls': module_cls, 'stashable_names': stashable_names, 'poppable_names': poppable_names}
    return type(name, bases, attrs)