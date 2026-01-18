from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
@property
def positional_values(self) -> List[LazyArgument]:
    return self.filtered_args(positional=True, keyword=False, values=True, scalars=False)