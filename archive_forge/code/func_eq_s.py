def eq_s(self, s):
    if self.limit - self.cursor < len(s):
        return False
    if self.current[self.cursor:self.cursor + len(s)] != s:
        return False
    self.cursor += len(s)
    return True