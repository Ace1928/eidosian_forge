from pythran.typing import List, Dict, Set, Fun, TypeVar
from pythran.typing import Union, Iterable
def extract_combiner(signature):
    if not isinstance(signature, (Fun, Union)):
        return None
    if type(signature) is Union:
        combiners = [extract_combiner(up) for up in signature.__args__]
        combiners = [cb for cb in combiners if cb]

        def combiner(self, node):
            for cb in combiners:
                cb(self, node)
        return combiner
    args = signature.__args__[:-1]
    if not args:
        return None
    deps = type_dependencies(args[0])
    if not deps:
        return None
    deps_src = dict()
    for i, arg in enumerate(args[1:]):
        arg_deps = type_dependencies(arg)
        common_deps = deps.intersection(arg_deps)
        for common_dep in common_deps:
            deps_src.setdefault(common_dep, []).append((i + 1, arg))
    return build_combiner(signature, deps_src)