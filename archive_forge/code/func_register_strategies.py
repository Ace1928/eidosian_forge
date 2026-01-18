import sys
from typing import Any
import lightning_fabric as fabric
from lightning_fabric.accelerators import XLAAccelerator
from lightning_fabric.plugins.precision import XLAPrecision
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.single_xla import SingleDeviceXLAStrategy
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation
@classmethod
def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
    if 'single_tpu' not in strategy_registry:
        strategy_registry.register('single_tpu', cls, description='Legacy class. Use `single_xla` instead.')