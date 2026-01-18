import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def record_shapeenv_event(*, save_tracked_fakes: bool=False) -> Callable:

    def decorator(fn: Callable) -> Callable:
        assert callable(fn)
        name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from torch.fx.experimental.symbolic_shapes import ShapeEnv
            if isinstance(args[0], ShapeEnv) and args[0].is_recording:
                return fn(*args, **kwargs)
            self = _extract_shape_env_and_assert_equal(args, kwargs)
            if self is None:
                return fn(*args, **kwargs)
            with self.recording():
                tracked_fakes = self.snapshot_tracked_fakes() if save_tracked_fakes else None
                event = ShapeEnvEvent(fn, list(args), kwargs, tracked_fakes, name=fn.__name__)
                self.events.append(event)
                return event.run(self)
        return wrapper
    return decorator