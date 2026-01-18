from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
@property
def base_name(self) -> str:
    return f'{self.name.name.base}'