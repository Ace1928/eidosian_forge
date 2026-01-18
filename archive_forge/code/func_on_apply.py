from __future__ import annotations
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, Callable
from .base import BasePool, apply_target
def on_apply(self, target: TargetFunction, args: tuple[Any, ...] | None=None, kwargs: dict[str, Any] | None=None, callback: Callable[..., Any] | None=None, accept_callback: Callable[..., Any] | None=None, **_: Any) -> ApplyResult:
    f = self.executor.submit(apply_target, target, args, kwargs, callback, accept_callback)
    return ApplyResult(f)