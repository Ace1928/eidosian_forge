@staticmethod
def combine(*args):
    baos = bytearray()
    for a in args:
        if type(a) in (list, bytearray, str, bytes):
            baos.extend(a)
        else:
            baos.append(a)
    return baos