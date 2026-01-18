def in_grouping_b(self, s, min, max):
    if self.cursor <= self.limit_backward:
        return False
    ch = ord(self.current[self.cursor - 1])
    if ch > max or ch < min:
        return False
    ch -= min
    if s[ch >> 3] & 1 << (ch & 7) == 0:
        return False
    self.cursor -= 1
    return True