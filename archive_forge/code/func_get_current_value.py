from typing import Optional
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.schedules.piecewise_schedule import PiecewiseSchedule
from ray.rllib.utils.typing import LearningRateOrSchedule, TensorType
def get_current_value(self) -> TensorType:
    """Returns the current value (as a tensor variable).

        This method should be used in loss functions of other (in-graph) places
        where the current value is needed.

        Returns:
            The tensor variable (holding the current value to be used).
        """
    return self._curr_value