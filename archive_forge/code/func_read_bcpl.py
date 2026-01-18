def read_bcpl(self, length):
    str_length = self.read_byte()
    data = self.f.read(length - 1)
    return data[:str_length]