from io import StringIO
def queryForString(self, elem):
    result = StringIO()
    self.baseLocation.queryForString(elem, result)
    return result.getvalue()