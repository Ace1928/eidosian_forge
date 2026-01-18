from dogpile.cache import api
from dogpile.cache import proxy
from oslo_context import context as oslo_context
from oslo_serialization import msgpackutils
def delete_multi(self, keys):
    for k in keys:
        self._delete_local_cache(k)
    self.proxied.delete_multi(keys)