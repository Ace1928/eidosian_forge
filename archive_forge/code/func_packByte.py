def packByte(self, v, n2):
    if v == 251:
        return self.packHex(n2)
    if v == 255:
        return self.packNibble(n2)
    return -1