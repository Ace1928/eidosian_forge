from collections import defaultdict
import networkx as nx
def dfs_orientation(self, v):
    """Orient the graph by DFS, compute lowpoints and nesting order."""
    dfs_stack = [v]
    ind = defaultdict(lambda: 0)
    skip_init = defaultdict(lambda: False)
    while dfs_stack:
        v = dfs_stack.pop()
        e = self.parent_edge[v]
        for w in self.adjs[v][ind[v]:]:
            vw = (v, w)
            if not skip_init[vw]:
                if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                    ind[v] += 1
                    continue
                self.DG.add_edge(v, w)
                self.lowpt[vw] = self.height[v]
                self.lowpt2[vw] = self.height[v]
                if self.height[w] is None:
                    self.parent_edge[w] = vw
                    self.height[w] = self.height[v] + 1
                    dfs_stack.append(v)
                    dfs_stack.append(w)
                    skip_init[vw] = True
                    break
                else:
                    self.lowpt[vw] = self.height[w]
            self.nesting_depth[vw] = 2 * self.lowpt[vw]
            if self.lowpt2[vw] < self.height[v]:
                self.nesting_depth[vw] += 1
            if e is not None:
                if self.lowpt[vw] < self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                    self.lowpt[e] = self.lowpt[vw]
                elif self.lowpt[vw] > self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                else:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])
            ind[v] += 1