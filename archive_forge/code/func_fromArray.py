from OpenGL._bytes import integer_types
def fromArray(cls, array, total):
    """Produce list with all records from the array"""
    result = []
    index = 0
    arrayLength = len(array)
    for item in range(total):
        if index + 2 >= arrayLength:
            break
        count = array[index]
        near = array[index + 1]
        far = array[index + 2]
        names = [uintToLong(v) for v in array[index + 3:index + 3 + count]]
        result.append(cls(near, far, names))
        index += 3 + count
    return result