from collections import defaultdict
from zope.interface.adapter import AdapterRegistry
from zope.interface.declarations import implementedBy
from zope.interface.declarations import implementer
from zope.interface.declarations import implementer_only
from zope.interface.declarations import providedBy
from zope.interface.interface import Interface
from zope.interface.interfaces import ComponentLookupError
from zope.interface.interfaces import IAdapterRegistration
from zope.interface.interfaces import IComponents
from zope.interface.interfaces import IHandlerRegistration
from zope.interface.interfaces import ISpecification
from zope.interface.interfaces import ISubscriptionAdapterRegistration
from zope.interface.interfaces import IUtilityRegistration
from zope.interface.interfaces import Registered
from zope.interface.interfaces import Unregistered
def rebuildUtilityRegistryFromLocalCache(self, rebuild=False):
    """
        Emergency maintenance method to rebuild the ``.utilities``
        registry from the local copy maintained in this object, or
        detect the need to do so.

        Most users will never need to call this, but it can be helpful
        in the event of suspected corruption.

        By default, this method only checks for corruption. To make it
        actually rebuild the registry, pass `True` for *rebuild*.

        :param bool rebuild: If set to `True` (not the default),
           this method will actually register and subscribe utilities
           in the registry as needed to synchronize with the local cache.

        :return: A dictionary that's meant as diagnostic data. The keys
           and values may change over time. When called with a false *rebuild*,
           the keys ``"needed_registered"`` and ``"needed_subscribed"`` will be
           non-zero if any corruption was detected, but that will not be corrected.

        .. versionadded:: 5.3.0
        """
    regs = dict(self._utility_registrations)
    utils = self.utilities
    needed_registered = 0
    did_not_register = 0
    needed_subscribed = 0
    did_not_subscribe = 0
    assert 'changed' not in utils.__dict__
    utils.changed = lambda _: None
    if rebuild:
        register = utils.register
        subscribe = utils.subscribe
    else:
        register = subscribe = lambda *args: None
    try:
        for (provided, name), (value, _info, _factory) in regs.items():
            if utils.registered((), provided, name) != value:
                register((), provided, name, value)
                needed_registered += 1
            else:
                did_not_register += 1
            if utils.subscribed((), provided, value) is None:
                needed_subscribed += 1
                subscribe((), provided, value)
            else:
                did_not_subscribe += 1
    finally:
        del utils.changed
        if rebuild and (needed_subscribed or needed_registered):
            utils.changed(utils)
    return {'needed_registered': needed_registered, 'did_not_register': did_not_register, 'needed_subscribed': needed_subscribed, 'did_not_subscribe': did_not_subscribe}