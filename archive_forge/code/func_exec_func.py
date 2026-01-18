from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
@staticmethod
def exec_func(fn: Callable, obj: Any, args: Tuple, kwargs: Dict) -> Any:
    """
        Execute the specified function.

        Parameters
        ----------
        fn : Callable
        obj : Any
        args : Tuple
        kwargs : dict

        Returns
        -------
        Any
        """
    try:
        try:
            return fn(obj, *args, **kwargs)
        except ValueError as err:
            if isinstance(obj, (pandas.DataFrame, pandas.Series)):
                return fn(obj.copy(), *args, **kwargs)
            else:
                raise err
    except Exception as err:
        get_logger().error(f'{err}. fn={fn}, obj={obj}, args={args}, kwargs={kwargs}')
        raise err