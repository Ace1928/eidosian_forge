from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
def store_invariants(dct, bases, destination_name, source_name):
    invariants = []
    for ns in [dct] + list(_all_dicts(bases)):
        try:
            invariant = ns[source_name]
        except KeyError:
            continue
        invariants.append(invariant)
    if not all((callable(invariant) for invariant in invariants)):
        raise TypeError('Invariants must be callable')
    dct[destination_name] = tuple((wrap_invariant(inv) for inv in invariants))