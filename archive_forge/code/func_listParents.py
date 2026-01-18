from io import StringIO
def listParents(self, elem, parentlist):
    if elem.parent != None:
        self.listParents(elem.parent, parentlist)
    parentlist.append(elem.name)