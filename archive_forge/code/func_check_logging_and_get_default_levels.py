from typing import Optional, Tuple, Union
from typing_extensions import TypedDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException
@classmethod
def check_logging_and_get_default_levels(cls, fx_name: str, on_step: Optional[bool], on_epoch: Optional[bool]) -> Tuple[bool, bool]:
    """Check if the given hook name is allowed to log and return logging levels."""
    cls.check_logging(fx_name)
    on_step, on_epoch = cls.get_default_logging_levels(fx_name, on_step, on_epoch)
    cls.check_logging_levels(fx_name, on_step, on_epoch)
    return (on_step, on_epoch)