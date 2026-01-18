from pythran.analyses.identifiers import Identifiers
from pythran.passmanager import NodeAnalysis
def check_comprehension(self, iters):
    targets = {gen.target.id for gen in iters}
    optimizable = True
    for it in iters:
        ids = self.gather(Identifiers, it)
        optimizable &= all(((ident == it.target.id) | (ident not in targets) for ident in ids))
    return optimizable