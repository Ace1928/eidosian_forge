def add_predicate(self, name, checker):
    self.predicate.append((checker, self.binding[name]))