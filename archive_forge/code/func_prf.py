def prf(msg, inner=inner, outer=outer):
    icpy = inner.copy()
    ocpy = outer.copy()
    icpy.update(msg)
    ocpy.update(icpy.digest())
    return ocpy.digest()