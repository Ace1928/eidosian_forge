from .functions import defun, defun_wrapped
def find_rosser_block_zero(ctx, n):
    """for n<400 000 000 determines a block were one find our zero"""
    for k in range(len(_ROSSER_EXCEPTIONS) // 2):
        a = _ROSSER_EXCEPTIONS[2 * k][0]
        b = _ROSSER_EXCEPTIONS[2 * k][1]
        if a <= n - 2 and n - 1 <= b:
            t0 = ctx.grampoint(a)
            t1 = ctx.grampoint(b)
            v0 = ctx._fp.siegelz(t0)
            v1 = ctx._fp.siegelz(t1)
            my_zero_number = n - a - 1
            zero_number_block = b - a
            pattern = _ROSSER_EXCEPTIONS[2 * k + 1]
            return (my_zero_number, [a, b], [t0, t1], [v0, v1])
    k = n - 2
    t, v, b = compute_triple_tvb(ctx, k)
    T = [t]
    V = [v]
    while b < 0:
        k -= 1
        t, v, b = compute_triple_tvb(ctx, k)
        T.insert(0, t)
        V.insert(0, v)
    my_zero_number = n - k - 1
    m = n - 1
    t, v, b = compute_triple_tvb(ctx, m)
    T.append(t)
    V.append(v)
    while b < 0:
        m += 1
        t, v, b = compute_triple_tvb(ctx, m)
        T.append(t)
        V.append(v)
    return (my_zero_number, [k, m], T, V)