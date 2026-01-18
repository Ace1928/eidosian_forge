from pythran.typing import List, Dict, Set, Fun, TypeVar
from pythran.typing import Union, Iterable
def dep_builder(type_var, ppal_index, index, t, self, node):
    if isinstance(t, TypeVar):
        if t is type_var:
            return lambda arg: arg if index == ppal_index else self.result[node.args[index]]
    elif isinstance(t, (List, Set, Iterable, Dict)):
        return lambda arg: self.builder.IteratorContentType(dep_builder(type_var, ppal_index, index, t.__args__[0], self, node)(arg))
    assert False, t