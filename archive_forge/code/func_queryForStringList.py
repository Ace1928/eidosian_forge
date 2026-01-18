from io import StringIO
def queryForStringList(self, elem):
    result = []
    self.baseLocation.queryForStringList(elem, result)
    if len(result) == 0:
        return None
    else:
        return result