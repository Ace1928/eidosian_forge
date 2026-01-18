import networkx as nx
def get_rows(self, r1, r2):
    for r in range(r1, r2 + 1):
        self.C[r % self.w, 1:] = self.solve_inverse(r)
    return self.C