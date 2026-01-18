from typing import Dict, List, Tuple
def parseModuliFile(filename: str) -> Dict[int, List[Tuple[int, int]]]:
    with open(filename) as f:
        lines = f.readlines()
    primes: Dict[int, List[Tuple[int, int]]] = {}
    for l in lines:
        l = l.strip()
        if not l or l[0] == '#':
            continue
        tim, typ, tst, tri, sizestr, genstr, modstr = l.split()
        size = int(sizestr) + 1
        gen = int(genstr)
        mod = int(modstr, 16)
        if size not in primes:
            primes[size] = []
        primes[size].append((gen, mod))
    return primes