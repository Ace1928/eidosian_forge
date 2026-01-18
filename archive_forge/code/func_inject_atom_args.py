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
def inject_atom_args(self, atom_name, pairs, transient=True):
    """Add values into storage for a specific atom only.

        :param transient: save the data in-memory only instead of persisting
                the data to backend storage (useful for resource-like objects
                or similar objects which can **not** be persisted)

        This method injects a dictionary/pairs of arguments for an atom so that
        when that atom is scheduled for execution it will have immediate access
        to these arguments.

        .. note::

            Injected atom arguments take precedence over arguments
            provided by predecessor atoms or arguments provided by injecting
            into the flow scope (using
            the :py:meth:`~taskflow.storage.Storage.inject` method).

        .. warning::

            It should be noted that injected atom arguments (that are scoped
            to the atom with the given name) *should* be serializable
            whenever possible. This is a **requirement** for the
            :doc:`worker based engine <workers>` which **must**
            serialize (typically using ``json``) all
            atom :py:meth:`~taskflow.atom.Atom.execute` and
            :py:meth:`~taskflow.atom.Atom.revert` arguments to
            be able to transmit those arguments to the target worker(s). If
            the use-case being applied/desired is to later use the worker
            based engine then it is highly recommended to ensure all injected
            atoms (even transient ones) are serializable to avoid issues
            that *may* appear later (when a object turned out to not actually
            be serializable).
        """
    if atom_name not in self._atom_name_to_uuid:
        raise exceptions.NotFound("Unknown atom name '%s'" % atom_name)

    def save_transient():
        self._injected_args.setdefault(atom_name, {})
        self._injected_args[atom_name].update(pairs)

    def save_persistent():
        source, clone = self._atomdetail_by_name(atom_name, clone=True)
        injected = source.meta.get(META_INJECTED)
        if not injected:
            injected = {}
        injected.update(pairs)
        clone.meta[META_INJECTED] = injected
        self._with_connection(self._save_atom_detail, source, clone)
    with self._lock.write_lock():
        if transient:
            save_transient()
        else:
            save_persistent()