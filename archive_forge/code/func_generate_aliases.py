from __future__ import annotations
import dataclasses
from typing import Callable, Literal
from ._internal import _internal_dataclass
def generate_aliases(self, field_name: str) -> tuple[str | None, str | AliasPath | AliasChoices | None, str | None]:
    """Generate `alias`, `validation_alias`, and `serialization_alias` for a field.

        Returns:
            A tuple of three aliases - validation, alias, and serialization.
        """
    alias = self._generate_alias('alias', (str,), field_name)
    validation_alias = self._generate_alias('validation_alias', (str, AliasChoices, AliasPath), field_name)
    serialization_alias = self._generate_alias('serialization_alias', (str,), field_name)
    return (alias, validation_alias, serialization_alias)