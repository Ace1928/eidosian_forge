import warnings
from pathlib import Path
from typing import Optional, Type, Union
from lightning_fabric.utilities.rank_zero import LightningDeprecationWarning
def disable_possible_user_warnings(module: str='') -> None:
    """Ignore warnings of the category ``PossibleUserWarning`` from Lightning.

    For more granular control over which warnings to ignore, use :func:`warnings.filterwarnings` directly.

    Args:
        module: Name of the module for which the warnings should be ignored (e.g., ``'pytorch_lightning.strategies'``).
            Default: Disables warnings from all modules.

    """
    warnings.filterwarnings('ignore', module=module, category=PossibleUserWarning)