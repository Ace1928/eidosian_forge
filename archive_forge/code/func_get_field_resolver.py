import inspect
from functools import partial
from itertools import chain
from wandb_promise import Promise
def get_field_resolver(self, field_resolver):
    if field_resolver not in self._cached_resolvers:
        self._cached_resolvers[field_resolver] = middleware_chain(field_resolver, self._middleware_resolvers, wrap_in_promise=self.wrap_in_promise)
    return self._cached_resolvers[field_resolver]