import typing as t
from . import nodes
from .visitor import NodeVisitor
def branch_update(self, branch_symbols: t.Sequence['Symbols']) -> None:
    stores: t.Dict[str, int] = {}
    for branch in branch_symbols:
        for target in branch.stores:
            if target in self.stores:
                continue
            stores[target] = stores.get(target, 0) + 1
    for sym in branch_symbols:
        self.refs.update(sym.refs)
        self.loads.update(sym.loads)
        self.stores.update(sym.stores)
    for name, branch_count in stores.items():
        if branch_count == len(branch_symbols):
            continue
        target = self.find_ref(name)
        assert target is not None, 'should not happen'
        if self.parent is not None:
            outer_target = self.parent.find_ref(name)
            if outer_target is not None:
                self.loads[target] = (VAR_LOAD_ALIAS, outer_target)
                continue
        self.loads[target] = (VAR_LOAD_RESOLVE, name)