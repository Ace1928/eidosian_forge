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
def fetch_mapped_args(self, args_mapping, atom_name=None, scope_walker=None, optional_args=None):
    """Fetch ``execute`` arguments for an atom using its args mapping."""

    def _extract_first_from(name, sources):
        """Extracts/returns first occurrence of key in list of dicts."""
        for i, source in enumerate(sources):
            if not source:
                continue
            if name in source:
                return (i, source[name])
        raise KeyError(name)
    if optional_args is None:
        optional_args = []
    if atom_name:
        source, _clone = self._atomdetail_by_name(atom_name)
        injected_sources = [self._injected_args.get(atom_name, {}), source.meta.get(META_INJECTED, {})]
        if scope_walker is None:
            scope_walker = self._scope_fetcher(atom_name)
    else:
        injected_sources = []
    if not args_mapping:
        return {}
    get_results = lambda atom_name: self._get(atom_name, 'last_results', 'failure', _EXECUTE_STATES_WITH_RESULTS, states.EXECUTE)
    mapped_args = {}
    for bound_name, name in args_mapping.items():
        if LOG.isEnabledFor(logging.TRACE):
            if atom_name:
                LOG.trace("Looking for %r <= %r for atom '%s'", bound_name, name, atom_name)
            else:
                LOG.trace('Looking for %r <= %r', bound_name, name)
        try:
            source_index, value = _extract_first_from(name, injected_sources)
            mapped_args[bound_name] = value
            if LOG.isEnabledFor(logging.TRACE):
                if source_index == 0:
                    LOG.trace('Matched %r <= %r to %r (from injected atom-specific transient values)', bound_name, name, value)
                else:
                    LOG.trace('Matched %r <= %r to %r (from injected atom-specific persistent values)', bound_name, name, value)
        except KeyError:
            try:
                maybe_providers = self._reverse_mapping[name]
            except KeyError:
                if bound_name in optional_args:
                    LOG.trace('Argument %r is optional, skipping', bound_name)
                    continue
                raise exceptions.NotFound('Name %r is not mapped as a produced output by any providers' % name)
            locator = _ProviderLocator(self._transients, functools.partial(self._fetch_providers, providers=maybe_providers), get_results)
            searched_providers, providers = locator.find(name, scope_walker=scope_walker)
            if not providers:
                raise exceptions.NotFound('Mapped argument %r <= %r was not produced by any accessible provider (%s possible providers were scanned)' % (bound_name, name, len(searched_providers)))
            provider, value = _item_from_first_of(providers, name)
            mapped_args[bound_name] = value
            LOG.trace('Matched %r <= %r to %r (from %s)', bound_name, name, value, provider)
    return mapped_args