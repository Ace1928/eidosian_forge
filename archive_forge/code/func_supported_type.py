from __future__ import annotations
from lightning_utilities.core.enums import StrEnum as LightningEnum
@staticmethod
def supported_type(val: str) -> bool:
    return any((x.value == val for x in GradClipAlgorithmType))