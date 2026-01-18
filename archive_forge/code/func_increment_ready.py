from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
def increment_ready(self) -> None:
    self.total.ready += 1
    self.current.ready += 1