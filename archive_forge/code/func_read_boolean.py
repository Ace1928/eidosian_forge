from struct import unpack
def read_boolean(self):
    """A boolean is written as a single byte whose value is either 0
        (false) or 1 (true).
        """
    return unpack('B', self.fo.read(1))[0] != 0