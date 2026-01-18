import math
def closest_prime(nt: int) -> int:
    if is_prime(nt):
        return nt
    lower = None
    higher = None
    for i in range(nt if nt % 2 != 0 else nt - 1, 1, -2):
        if is_prime(i):
            lower = i
            break
    c = nt + 1
    while higher is None:
        if is_prime(c):
            higher = c
        else:
            c += 2 if c % 2 != 0 else 1
    return higher if lower is None or higher - nt < nt - lower else lower