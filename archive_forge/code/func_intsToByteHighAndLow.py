@staticmethod
def intsToByteHighAndLow(highValue, lowValue):
    highValue = ord(highValue) if type(highValue) is str else highValue
    lowValue = ord(lowValue) if type(lowValue) is str else lowValue
    return ((highValue << 4 | lowValue) & 255) % 256