from io import StringIO
def matchesPredicates(self, elem):
    for p in self.predicates:
        if not p.value(elem):
            return 0
    return 1