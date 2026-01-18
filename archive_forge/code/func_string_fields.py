from typing import Any, Dict, Optional, Tuple
from wandb.data_types import Table
from wandb.errors import Error
@property
def string_fields(self) -> Dict[str, Any]:
    return self._string_fields