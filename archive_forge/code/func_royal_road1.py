from functools import wraps
def royal_road1(individual, order):
    """Royal Road Function R1 as presented by Melanie Mitchell in :
    "An introduction to Genetic Algorithms".
    """
    nelem = len(individual) // order
    max_value = int(2 ** order - 1)
    total = 0
    for i in range(nelem):
        value = int(''.join(map(str, individual[i * order:i * order + order])), 2)
        total += int(order) * int(value / max_value)
    return (total,)