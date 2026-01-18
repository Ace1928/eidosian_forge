import inspect
from ray._private.utils import get_function_args
from ray.tune.schedulers.trial_scheduler import TrialScheduler, FIFOScheduler
from ray.tune.schedulers.hyperband import HyperBandScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune.schedulers.median_stopping_rule import MedianStoppingRule
from ray.tune.schedulers.pbt import (
from ray.tune.schedulers.resource_changing_scheduler import ResourceChangingScheduler
from ray.util import PublicAPI
@PublicAPI(stability='beta')
def create_scheduler(scheduler, **kwargs):
    """Instantiate a scheduler based on the given string.

    This is useful for swapping between different schedulers.

    Args:
        scheduler: The scheduler to use.
        **kwargs: Scheduler parameters.
            These keyword arguments will be passed to the initialization
            function of the chosen scheduler.
    Returns:
        ray.tune.schedulers.trial_scheduler.TrialScheduler: The scheduler.
    Example:
        >>> from ray import tune
        >>> pbt_kwargs = {}
        >>> scheduler = tune.create_scheduler('pbt', **pbt_kwargs) # doctest: +SKIP
    """
    scheduler = scheduler.lower()
    if scheduler not in SCHEDULER_IMPORT:
        raise ValueError(f'The `scheduler` argument must be one of {list(SCHEDULER_IMPORT)}. Got: {scheduler}')
    SchedulerClass = SCHEDULER_IMPORT[scheduler]
    if inspect.isfunction(SchedulerClass):
        SchedulerClass = SchedulerClass()
    scheduler_args = get_function_args(SchedulerClass)
    trimmed_kwargs = {k: v for k, v in kwargs.items() if k in scheduler_args}
    return SchedulerClass(**trimmed_kwargs)