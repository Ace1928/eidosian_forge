import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
@DeveloperAPI
def diagnose_serialization(trainable: Callable):
    """Utility for detecting why your trainable function isn't serializing.

    Args:
        trainable: The trainable object passed to
            tune.Tuner(trainable). Currently only supports
            Function API.

    Returns:
        bool | set of unserializable objects.

    Example:

    .. code-block:: python

        import threading
        # this is not serializable
        e = threading.Event()

        def test():
            print(e)

        diagnose_serialization(test)
        # should help identify that 'e' should be moved into
        # the `test` scope.

        # correct implementation
        def test():
            e = threading.Event()
            print(e)

        assert diagnose_serialization(test) is True

    """
    from ray.tune.registry import register_trainable, _check_serializability

    def check_variables(objects, failure_set, printer):
        for var_name, variable in objects.items():
            msg = None
            try:
                _check_serializability(var_name, variable)
                status = 'PASSED'
            except Exception as e:
                status = 'FAILED'
                msg = f'{e.__class__.__name__}: {str(e)}'
                failure_set.add(var_name)
            printer(f"{str(variable)}[name='{var_name}'']... {status}")
            if msg:
                printer(msg)
    print(f'Trying to serialize {trainable}...')
    try:
        register_trainable('__test:' + str(trainable), trainable, warn=False)
        print('Serialization succeeded!')
        return True
    except Exception as e:
        print(f'Serialization failed: {e}')
    print(f'Inspecting the scope of the trainable by running `inspect.getclosurevars({str(trainable)})`...')
    closure = inspect.getclosurevars(trainable)
    failure_set = set()
    if closure.globals:
        print(f'Detected {len(closure.globals)} global variables. Checking serializability...')
        check_variables(closure.globals, failure_set, lambda s: print('   ' + s))
    if closure.nonlocals:
        print(f'Detected {len(closure.nonlocals)} nonlocal variables. Checking serializability...')
        check_variables(closure.nonlocals, failure_set, lambda s: print('   ' + s))
    if not failure_set:
        print('Nothing was found to have failed the diagnostic test, though serialization did not succeed. Feel free to raise an issue on github.')
        return failure_set
    else:
        print(f'Variable(s) {failure_set} was found to be non-serializable. Consider either removing the instantiation/imports of these objects or moving them into the scope of the trainable. ')
        return failure_set