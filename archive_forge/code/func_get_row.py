import networkx as nx
def get_row(self, r):
    self.C[r % self.w, 1:] = self.solve_inverse(r)
    return self.C[r % self.w]