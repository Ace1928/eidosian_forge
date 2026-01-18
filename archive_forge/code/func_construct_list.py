from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
@classmethod
def construct_list(cls, lst: List, chain: List, refs: Dict[int, Any], meta: List):
    """
        Construct the list.

        Parameters
        ----------
        lst : list
        chain : list
        refs : dict
        meta : list

        Yields
        ------
        Any
            Either ``construct_chain()`` or ``construct_list()`` generator.
        """
    pop = chain.pop
    lst_append = lst.append
    while True:
        obj = pop()
        if isinstance(obj, _Tag):
            if obj == _Tag.END:
                break
            elif obj == _Tag.CHAIN:
                yield cls.construct_chain(chain, refs, meta, lst)
            elif obj == _Tag.LIST:
                lst_append([])
                yield cls.construct_list(lst[-1], chain, refs, meta)
            elif obj is _Tag.REF:
                lst_append(refs[pop()])
            else:
                raise ValueError(f'Unexpected tag {obj}')
        else:
            lst_append(obj)