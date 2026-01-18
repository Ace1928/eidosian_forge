import math
def prime_factors(n) -> list[int]:
    fs = []
    if n <= 1:
        return fs
    while n % 2 == 0:
        fs.append(2)
        n = n // 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            fs.append(i)
            n = n / i
    if n > 2:
        fs.append(n)
    return fs