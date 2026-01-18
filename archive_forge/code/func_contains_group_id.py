from __future__ import print_function
import os
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from .utils import make_dirs_if_not_exists
def contains_group_id(self, group_id):
    """
        Args:
            :param group_id: (int) The number of group we want to check.
            :return: True if fold contains line or lines with that group id.

        """
    return group_id in self._fold