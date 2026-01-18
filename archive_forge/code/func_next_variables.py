from typing import List, Optional
from ..exc import unimplemented
from .base import VariableTracker
from .constant import ConstantVariable
def next_variables(self, tx):
    assert self.mutable_local
    if self.iterator is not None:
        try:
            new_item, next_inner_iter = self.iterator.next_variables(tx)
            tx.replace_all(self.iterator, next_inner_iter)
            if len(self.saved) > MAX_CYCLE:
                unimplemented('input iterator to itertools.cycle has too many items')
            next_iter = self.clone(iterator=next_inner_iter, saved=self.saved + [new_item], item=new_item)
            tx.replace_all(self, next_iter)
            if self.item is None:
                return next_iter.next_variables(tx)
            return (self.item, next_iter)
        except StopIteration:
            next_iter = self.clone(iterator=None)
            tx.replace_all(self, next_iter)
            return next_iter.next_variables(tx)
    elif len(self.saved) > 0:
        next_iter = self.clone(saved_index=(self.saved_index + 1) % len(self.saved), item=self.saved[self.saved_index])
        tx.replace_all(self, next_iter)
        return (self.item, next_iter)
    else:
        raise StopIteration
    return (self.item, next_iter)