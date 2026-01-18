from types import FunctionType
import sys
def minimalBases(classes):
    """Reduce a list of base classes to its ordered minimum equivalent"""
    candidates = []
    for m in classes:
        for n in classes:
            if issubclass(n, m) and m is not n:
                break
        else:
            if m in candidates:
                candidates.remove(m)
            candidates.append(m)
    return candidates