from io import StringIO
def queryForNodes(self, elem):
    result = []
    self.baseLocation.queryForNodes(elem, result)
    if len(result) == 0:
        return None
    else:
        return result