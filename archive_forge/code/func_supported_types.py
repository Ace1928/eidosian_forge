from __future__ import annotations
from lightning_utilities.core.enums import StrEnum as LightningEnum
@staticmethod
def supported_types() -> list[str]:
    return [x.value for x in GradClipAlgorithmType]