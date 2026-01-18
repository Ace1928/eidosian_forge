import sys
def candidate_pairs_iter(self):
    """Iterator over candidate pairs of nodes in G1 and G2."""
    G1_nodes = self.G1_nodes
    G2_nodes = self.G2_nodes
    min_key = self.G2_node_order.__getitem__
    T1_out = [node for node in self.out_1 if node not in self.core_1]
    T2_out = [node for node in self.out_2 if node not in self.core_2]
    if T1_out and T2_out:
        node_2 = min(T2_out, key=min_key)
        for node_1 in T1_out:
            yield (node_1, node_2)
    else:
        T1_in = [node for node in self.in_1 if node not in self.core_1]
        T2_in = [node for node in self.in_2 if node not in self.core_2]
        if T1_in and T2_in:
            node_2 = min(T2_in, key=min_key)
            for node_1 in T1_in:
                yield (node_1, node_2)
        else:
            node_2 = min(G2_nodes - set(self.core_2), key=min_key)
            for node_1 in G1_nodes:
                if node_1 not in self.core_1:
                    yield (node_1, node_2)