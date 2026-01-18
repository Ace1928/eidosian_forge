from OpenGL._bytes import integer_types
def convertDistance(self, value):
    """Convert a distance value from array uint to 0.0-1.0 range float"""
    return uintToLong(value) / self.DISTANCE_DIVISOR