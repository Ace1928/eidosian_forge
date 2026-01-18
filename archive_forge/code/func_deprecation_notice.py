from collections.abc import Iterable, Sequence
import wandb
from wandb import util
from wandb.sdk.lib import deprecate
def deprecation_notice() -> None:
    deprecate.deprecate(field_name=deprecate.Deprecated.plots, warning_message='wandb.plots.* functions are deprecated and will be removed in a future release. Please use wandb.plot.* instead.')