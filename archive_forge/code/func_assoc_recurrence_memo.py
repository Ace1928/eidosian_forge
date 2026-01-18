from functools import wraps
def assoc_recurrence_memo(base_seq):
    """
    Memo decorator for associated sequences defined by recurrence starting from base

    base_seq(n) -- callable to get base sequence elements

    XXX works only for Pn0 = base_seq(0) cases
    XXX works only for m <= n cases
    """
    cache = []

    def decorator(f):

        @wraps(f)
        def g(n, m):
            L = len(cache)
            if n < L:
                return cache[n][m]
            for i in range(L, n + 1):
                F_i0 = base_seq(i)
                F_i_cache = [F_i0]
                cache.append(F_i_cache)
                for j in range(1, i + 1):
                    F_ij = f(i, j, cache)
                    F_i_cache.append(F_ij)
            return cache[n][m]
        return g
    return decorator