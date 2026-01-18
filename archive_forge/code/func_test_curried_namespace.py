import toolz
import toolz.curried
from toolz.curried import (take, first, second, sorted, merge_with, reduce,
from collections import defaultdict
from importlib import import_module
from operator import add
def test_curried_namespace():
    exceptions = import_module('toolz.curried.exceptions')
    namespace = {}

    def curry_namespace(ns):
        return {name: toolz.curry(f) if should_curry(f) else f for name, f in ns.items() if '__' not in name}
    from_toolz = curry_namespace(vars(toolz))
    from_exceptions = curry_namespace(vars(exceptions))
    namespace.update(toolz.merge(from_toolz, from_exceptions))
    namespace = toolz.valfilter(callable, namespace)
    curried_namespace = toolz.valfilter(callable, toolz.curried.__dict__)
    if namespace != curried_namespace:
        missing = set(namespace) - set(curried_namespace)
        if missing:
            raise AssertionError('There are missing functions in toolz.curried:\n    %s' % '    \n'.join(sorted(missing)))
        extra = set(curried_namespace) - set(namespace)
        if extra:
            raise AssertionError('There are extra functions in toolz.curried:\n    %s' % '    \n'.join(sorted(extra)))
        unequal = toolz.merge_with(list, namespace, curried_namespace)
        unequal = toolz.valfilter(lambda x: x[0] != x[1], unequal)
        messages = []
        for name, (orig_func, auto_func) in sorted(unequal.items()):
            if name in from_exceptions:
                messages.append('%s should come from toolz.curried.exceptions' % name)
            elif should_curry(getattr(toolz, name)):
                messages.append('%s should be curried from toolz' % name)
            else:
                messages.append('%s should come from toolz and NOT be curried' % name)
        raise AssertionError('\n'.join(messages))