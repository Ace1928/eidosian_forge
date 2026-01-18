from io import StringIO
def findFirstRootMatch(self, elem):
    if (self.elementName == None or self.elementName == elem.name) and self.matchesPredicates(elem):
        if self.childLocation != None:
            for c in elem.elements():
                if self.childLocation.matches(c):
                    return c
            return None
        else:
            return elem
    else:
        for c in elem.elements():
            if self.matches(c):
                return c
        return None