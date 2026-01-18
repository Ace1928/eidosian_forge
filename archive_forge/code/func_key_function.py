from __future__ import annotations
import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable
def key_function(individual_stat):
    return (individual_stat.category_name, individual_stat.cache_name)