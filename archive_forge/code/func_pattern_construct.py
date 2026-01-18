from .functions import defun, defun_wrapped
def pattern_construct(ctx, block, T, V):
    pattern = '('
    a = block[0]
    b = block[1]
    t0, v0, b0 = compute_triple_tvb(ctx, a)
    k = 0
    k0 = 0
    for n in range(a + 1, b + 1):
        t1, v1, b1 = compute_triple_tvb(ctx, n)
        lgT = len(T)
        while k < lgT and T[k] <= t1:
            k += 1
        L = V[k0:k]
        L.append(v1)
        L.insert(0, v0)
        count = count_variations(L)
        pattern = pattern + '%s' % count
        if b1 > 0:
            pattern = pattern + ')('
        k0 = k
        t0, v0, b0 = (t1, v1, b1)
    pattern = pattern[:-1]
    return pattern