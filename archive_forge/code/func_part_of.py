import shelve
from repoze.who.interfaces import IMetadataProvider
from zope.interface import implements
def part_of(self, user, virtualorg):
    if virtualorg in self._store[user]['entitlement']:
        return True
    else:
        return False