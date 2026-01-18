import json
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence
import requests
@property
def get_schema_elements(self) -> Dict[str, Any]:
    return self.schema_elements