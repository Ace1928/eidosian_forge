import abc
from yaql.language import exceptions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
def collect_functions(self, name, predicate=None, use_convention=False):
    overloads = []
    p = self
    while p is not None:
        context_predicate = None
        if predicate:
            context_predicate = lambda fd: predicate(fd, p)
        layer_overloads, is_exclusive = p.get_functions(name, context_predicate, use_convention)
        p = None if is_exclusive else p.parent
        if layer_overloads:
            overloads.append(layer_overloads)
    return overloads