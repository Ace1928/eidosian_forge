from typing import Any, Dict, Optional
from uuid import uuid4
from tune.concepts.flow.report import TrialReport
def create_child(self, name: Optional[str]=None, description: Optional[str]=None, is_step: bool=False) -> 'MetricLogger':
    """Create a child logger

        :param name: the name of the child logger, defaults to None
        :param description: the long description of the child logger, defaults to None
        :param is_step: whether the child logger is a sub-step, for example for
            an epoch of a deep learning model, it should set to ``True``

        :return: the child logger
        """
    return self