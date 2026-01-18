from inspect import isclass
def instanciate(self, caller, arguments):
    if self.fun is caller:
        return builder.UnknownType
    else:
        return InstantiatedType(self.fun, self.name, arguments)