import shelve
from repoze.who.interfaces import IMetadataProvider
from zope.interface import implements
def store_entitlement(self, user, virtualorg, entitlement=None):
    if user not in self._store:
        self._store[user] = {'entitlement': {}}
    elif 'entitlement' not in self._store[user]:
        self._store[user]['entitlement'] = {}
    if entitlement is None:
        entitlement = []
    self._store[user]['entitlement'][virtualorg] = entitlement
    self._store.sync()