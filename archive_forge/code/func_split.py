@staticmethod
def split(inp, firstLength, secondLength, thirdLength=None):
    parts = []
    parts.append(inp[:firstLength])
    parts.append(inp[firstLength:firstLength + secondLength])
    if thirdLength is not None:
        parts.append(inp[firstLength + secondLength:firstLength + secondLength + thirdLength])
    return parts