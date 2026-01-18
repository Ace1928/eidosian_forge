import logging
import warnings
from collections import OrderedDict
from typing import Union, Iterable, Dict
import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.utils as utils
def average_parameters(self, params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]]):
    """
        Averages parameters or parameter groups of an optimizer if ``step`` is no less than ``warmup_steps``
        and it can be divided by a period in the keys of ``period_process_group_dict``,
        where ``step`` is increased by 1 at each iteration in the training loop.
        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        only the largest period is used, and the corresponding process group is used for averaging parameters.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.
        """
    if self.step >= self.warmup_steps:
        group = self._find_process_group()
        if group is not None:
            utils.average_parameters_or_parameter_groups(params, group)
    self.step += 1