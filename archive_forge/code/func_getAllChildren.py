import binascii
def getAllChildren(self, tag=None):
    ret = []
    if tag is None:
        return self.children
    for c in self.children:
        if tag == c.tag:
            ret.append(c)
    return ret