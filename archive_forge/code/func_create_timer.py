import logging
import asyncio
import contextvars
import inspect
from collections import deque
from functools import partial, reduce
import copy
from ..core import State, Condition, Transition, EventData, listify
from ..core import Event, MachineError, Machine
from .nesting import HierarchicalMachine, NestedState, NestedEvent, NestedTransition, resolve_order
def create_timer(self, event_data):
    """
        Creates and returns a running timer. Shields self._process_timeout to prevent cancellation when
        transitioning away from the current state (which cancels the timer) while processing timeout callbacks.
        Args:
            event_data (EventData): Data representing the currently processed event.

        Returns (cancellable): A running timer with a cancel method
        """

    async def _timeout():
        try:
            await asyncio.sleep(self.timeout)
            await asyncio.shield(self._process_timeout(event_data))
        except asyncio.CancelledError:
            pass
    return asyncio.ensure_future(_timeout())