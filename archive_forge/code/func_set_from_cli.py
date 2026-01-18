import dataclasses
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
def set_from_cli(self, cli_value: str) -> None:
    if not cli_value.strip():
        self._value = []
    else:
        self._value = [item.strip() for item in cli_value.split(',')]