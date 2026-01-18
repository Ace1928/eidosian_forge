import enum
import timeit
import textwrap
from typing import overload, Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface
def blocked_autorange(self, callback: Optional[Callable[[int, float], NoReturn]]=None, min_run_time: float=0.2) -> common.Measurement:
    """Measure many replicates while keeping timer overhead to a minimum.

        At a high level, blocked_autorange executes the following pseudo-code::

            `setup`

            total_time = 0
            while total_time < min_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                total_time += (timer() - start)

        Note the variable `block_size` in the inner loop. The choice of block
        size is important to measurement quality, and must balance two
        competing objectives:

            1) A small block size results in more replicates and generally
               better statistics.

            2) A large block size better amortizes the cost of `timer`
               invocation, and results in a less biased measurement. This is
               important because CUDA synchronization time is non-trivial
               (order single to low double digit microseconds) and would
               otherwise bias the measurement.

        blocked_autorange sets block_size by running a warmup period,
        increasing block size until timer overhead is less than 0.1% of
        the overall computation. This value is then used for the main
        measurement loop.

        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
        """
    number = self._estimate_block_size(min_run_time)

    def time_hook() -> float:
        return self._timeit(number)

    def stop_hook(times: List[float]) -> bool:
        return True
    times = self._threaded_measurement_loop(number, time_hook, stop_hook, min_run_time=min_run_time, callback=callback)
    return common.Measurement(number_per_run=number, raw_times=times, task_spec=self._task_spec)