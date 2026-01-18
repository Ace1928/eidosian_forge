from pythran.typing import List, Dict, Set, Fun, TypeVar
from pythran.typing import Union, Iterable
def path_to(self, t, deps_builders, node):
    if isinstance(t, TypeVar):
        if t in deps_builders:
            return deps_builders[t]
        else:
            raise InfeasibleCombiner()
    if isinstance(t, List):
        return lambda arg: self.builder.ListType(path_to(self, t.__args__[0], deps_builders, node)(arg))
    if isinstance(t, Set):
        return lambda arg: self.builder.SetType(path_to(self, t.__args__[0], deps_builders, node)(arg))
    if isinstance(t, Dict):
        return lambda arg: self.builder.DictType(path_to(self, t.__args__[0], deps_builders, node)(arg), path_to(self, t.__args__[1], deps_builders, node)(arg))
    if isinstance(t, Fun):
        raise InfeasibleCombiner()
    if isinstance(t, Iterable):
        raise InfeasibleCombiner()
    assert False, (t, t.mro())