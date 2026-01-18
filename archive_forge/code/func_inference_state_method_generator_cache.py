from functools import wraps
from jedi import debug
def inference_state_method_generator_cache():
    """
    This is a special memoizer. It memoizes generators and also checks for
    recursion errors and returns no further iterator elemends in that case.
    """

    def func(function):

        @wraps(function)
        def wrapper(obj, *args, **kwargs):
            cache = obj.inference_state.memoize_cache
            try:
                memo = cache[function]
            except KeyError:
                cache[function] = memo = {}
            key = (obj, args, frozenset(kwargs.items()))
            if key in memo:
                actual_generator, cached_lst = memo[key]
            else:
                actual_generator = function(obj, *args, **kwargs)
                cached_lst = []
                memo[key] = (actual_generator, cached_lst)
            i = 0
            while True:
                try:
                    next_element = cached_lst[i]
                    if next_element is _RECURSION_SENTINEL:
                        debug.warning('Found a generator recursion for %s' % obj)
                        return
                except IndexError:
                    cached_lst.append(_RECURSION_SENTINEL)
                    next_element = next(actual_generator, None)
                    if next_element is None:
                        cached_lst.pop()
                        return
                    cached_lst[-1] = next_element
                yield next_element
                i += 1
        return wrapper
    return func