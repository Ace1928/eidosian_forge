def find_among_b(self, v):
    """
        find_among_b is for backwards processing. Same comments apply
        """
    i = 0
    j = len(v)
    c = self.cursor
    lb = self.limit_backward
    common_i = 0
    common_j = 0
    first_key_inspected = False
    while True:
        k = i + (j - i >> 1)
        diff = 0
        common = min(common_i, common_j)
        w = v[k]
        for i2 in range(len(w.s) - 1 - common, -1, -1):
            if c - common == lb:
                diff = -1
                break
            diff = ord(self.current[c - 1 - common]) - ord(w.s[i2])
            if diff != 0:
                break
            common += 1
        if diff < 0:
            j = k
            common_j = common
        else:
            i = k
            common_i = common
        if j - i <= 1:
            if i > 0:
                break
            if j == i:
                break
            if first_key_inspected:
                break
            first_key_inspected = True
    while True:
        w = v[i]
        if common_i >= len(w.s):
            self.cursor = c - len(w.s)
            if w.method is None:
                return w.result
            method = getattr(self, w.method)
            res = method()
            self.cursor = c - len(w.s)
            if res:
                return w.result
        i = w.substring_i
        if i < 0:
            return 0
    return -1