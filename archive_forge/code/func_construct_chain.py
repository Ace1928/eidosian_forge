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
def construct_chain(cls, chain: List, refs: Dict[int, Any], meta: List, lst: Optional[List]):
    """
        Construct the chain and execute it one by one.

        Parameters
        ----------
        chain : list
            A flat list containing the execution tree, deconstructed by
            ``DeferredExecution._deconstruct()``.
        refs : dict
            If an execution result is required for multiple chains, the
            reference to this result is saved in this dict.
        meta : list
            The lengths of the returned objects are added to this list.
        lst : list
            If specified, the execution result is added to this list.
            This is used when a chain is passed as an argument to a
            DeferredExecution task.

        Yields
        ------
        Any
            Either the ``construct_list()`` generator or the execution results.
        """
    pop = chain.pop
    tg_e = _Tag.END
    obj = pop()
    if obj is _Tag.REF:
        obj = refs[pop()]
    elif obj is _Tag.LIST:
        obj = []
        yield cls.construct_list(obj, chain, refs, meta)
    while chain:
        fn = pop()
        if fn == tg_e:
            lst.append(obj)
            break
        if (args_len := pop()) >= 0:
            if args_len == 0:
                args = []
            else:
                args = chain[-args_len:]
                del chain[-args_len:]
                args.reverse()
        else:
            args = []
            yield cls.construct_list(args, chain, refs, meta)
        if (args_len := pop()) >= 0:
            kwargs = {pop(): pop() for _ in range(args_len)}
        else:
            values = []
            yield cls.construct_list(values, chain, refs, meta)
            kwargs = {pop(): v for v in values}
        obj = cls.exec_func(fn, obj, args, kwargs)
        if (ref := pop()):
            refs[ref] = obj
        if (num_returns := pop()) == 0:
            continue
        itr = iter([obj] if num_returns == 1 else obj)
        for _ in range(num_returns):
            obj = next(itr)
            meta.append(len(obj) if hasattr(obj, '__len__') else 0)
            meta.append(len(obj.columns) if hasattr(obj, 'columns') else 0)
            yield obj