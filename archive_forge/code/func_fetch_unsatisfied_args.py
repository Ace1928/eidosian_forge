import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
@fasteners.read_locked
def fetch_unsatisfied_args(self, atom_name, args_mapping, scope_walker=None, optional_args=None):
    """Fetch unsatisfied ``execute`` arguments using an atoms args mapping.

        NOTE(harlowja): this takes into account the provided scope walker
        atoms who should produce the required value at runtime, as well as
        the transient/persistent flow and atom specific injected arguments.
        It does **not** check if the providers actually have produced the
        needed values; it just checks that they are registered to produce
        it in the future.
        """
    source, _clone = self._atomdetail_by_name(atom_name)
    if scope_walker is None:
        scope_walker = self._scope_fetcher(atom_name)
    if optional_args is None:
        optional_args = []
    injected_sources = [self._injected_args.get(atom_name, {}), source.meta.get(META_INJECTED, {})]
    missing = set(args_mapping.keys())
    locator = _ProviderLocator(self._transients, self._fetch_providers, lambda atom_name: self._get(atom_name, 'last_results', 'failure', _EXECUTE_STATES_WITH_RESULTS, states.EXECUTE))
    for bound_name, name in args_mapping.items():
        if LOG.isEnabledFor(logging.TRACE):
            LOG.trace("Looking for %r <= %r for atom '%s'", bound_name, name, atom_name)
        if bound_name in optional_args:
            LOG.trace('Argument %r is optional, skipping', bound_name)
            missing.discard(bound_name)
            continue
        maybe_providers = 0
        for source in injected_sources:
            if not source:
                continue
            if name in source:
                maybe_providers += 1
        maybe_providers += len(locator.find_potentials(name, scope_walker=scope_walker))
        if maybe_providers:
            LOG.trace("Atom '%s' will have %s potential providers of %r <= %r", atom_name, maybe_providers, bound_name, name)
            missing.discard(bound_name)
    return missing