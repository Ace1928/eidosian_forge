from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
@contextlib.contextmanager
def hold_trait_notifications(self) -> t.Any:
    """Context manager for bundling trait change notifications and cross
        validation.

        Use this when doing multiple trait assignments (init, config), to avoid
        race conditions in trait notifiers requesting other trait values.
        All trait notifications will fire after all values have been assigned.
        """
    if self._cross_validation_lock:
        yield
        return
    else:
        cache: dict[str, list[Bunch]] = {}

        def compress(past_changes: list[Bunch] | None, change: Bunch) -> list[Bunch]:
            """Merges the provided change with the last if possible."""
            if past_changes is None:
                return [change]
            else:
                if past_changes[-1]['type'] == 'change' and change.type == 'change':
                    past_changes[-1]['new'] = change.new
                else:
                    past_changes.append(change)
                return past_changes

        def hold(change: Bunch) -> None:
            name = change.name
            cache[name] = compress(cache.get(name), change)
        try:
            self.notify_change = hold
            self._cross_validation_lock = True
            yield
            for name in list(cache.keys()):
                trait = getattr(self.__class__, name)
                value = trait._cross_validate(self, getattr(self, name))
                self.set_trait(name, value)
        except TraitError as e:
            self.notify_change = lambda x: None
            for name, changes in cache.items():
                for change in changes[::-1]:
                    if change.type == 'change':
                        if change.old is not Undefined:
                            self.set_trait(name, change.old)
                        else:
                            self._trait_values.pop(name)
            cache = {}
            raise e
        finally:
            self._cross_validation_lock = False
            del self.notify_change
            for changes in cache.values():
                for change in changes:
                    self.notify_change(change)