import networkx as nx
def init_solver(self, L):
    global sp
    import scipy as sp
    ilu = sp.sparse.linalg.spilu(self.L1.tocsc())
    n = self.n - 1
    self.M = sp.sparse.linalg.LinearOperator(shape=(n, n), matvec=ilu.solve)