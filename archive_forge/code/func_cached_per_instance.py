import functools
import weakref
import numpy as np
from tensorflow.python.util import nest
def cached_per_instance(f):
    """Lightweight decorator for caching lazily constructed properties.

  When to use:
  This decorator provides simple caching with minimal overhead. It is designed
  for properties which are expensive to compute and static over the life of a
  class instance, and provides no mechanism for cache invalidation. Thus it is
  best suited for lazily exposing derived properties of other static data.

  For classes with custom getattr / setattr behavior (such as trackable
  objects), storing cache results as object attributes is not performant.
  Instead, a specialized cache can significantly reduce property lookup
  overhead. (While still allowing the decorated property to be lazily computed.)
  Consider the following class:

  ```
  class MyClass(object):
    def __setattr__(self, key, value):
      # Some expensive class specific code
      # ...
      # ...

      super(MyClass, self).__setattr__(key, value)

    @property
    def thing(self):
      # `thing` is expensive to compute (and may not even be requested), so we
      # want to lazily compute it and then cache it.
      output = getattr(self, '_thing', None)
      if output is None:
        self._thing = output = compute_thing(self)
      return output
  ```

  It's also worth noting that ANY overriding of __setattr__, even something as
  simple as:
  ```
    def __setattr__(self, key, value):
      super(MyClass, self).__setattr__(key, value)
  ```

  Slows down attribute assignment by nearly 10x.

  By contrast, replacing the definition of `thing` with the following sidesteps
  the expensive __setattr__ altogether:

  '''
  @property
  @tracking.cached_per_instance
  def thing(self):
    # `thing` is expensive to compute (and may not even be requested), so we
    # want to lazily compute it and then cache it.
    return compute_thing(self)
  '''

  Performance:
  The overhead for this decorator is ~0.4 us / call. A much lower overhead
  implementation (~0.085 us / call) can be achieved by using a custom dict type:

  ```
  def dict_based_cache(f):
    class Cache(dict):
      __slots__ = ()
      def __missing__(self, key):
        self[key] = output = f(key)
        return output

    return property(Cache().__getitem__)
  ```

  However, that implementation holds class instances as keys, and as a result
  blocks garbage collection. (And modifying it to use weakref's as keys raises
  the lookup overhead to ~0.4 us) As a result, the WeakKeyDictionary
  implementation below turns out to be more prudent.

  Args:
    f: The function to cache.

  Returns:
    f decorated with simple caching behavior.
  """
    cache = weakref.WeakKeyDictionary()

    @functools.wraps(f)
    def wrapped(item):
        output = cache.get(item)
        if output is None:
            cache[item] = output = f(item)
        return output
    wrapped.cache = cache
    return wrapped