def packNibble(self, n):
    if n in (45, 46):
        return 10 + (n - 45)
    if n in range(48, 58):
        return n - 48
    return -1