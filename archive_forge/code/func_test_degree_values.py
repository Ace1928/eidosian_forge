from pytest import approx
from networkx import is_connected, neighbors
from networkx.generators.internet_as_graphs import random_internet_as_graph
def test_degree_values(self):
    d_m = 0
    d_cp = 0
    d_c = 0
    p_m_m = 0
    p_cp_m = 0
    p_cp_cp = 0
    t_m = 0
    t_cp = 0
    t_c = 0
    for i, j in self.G.edges():
        e = self.G.edges[i, j]
        if e['type'] == 'transit':
            cust = int(e['customer'])
            if i == cust:
                prov = j
            elif j == cust:
                prov = i
            else:
                raise ValueError('Inconsistent data in the graph edge attributes')
            if cust in self.M:
                d_m += 1
                if self.G.nodes[prov]['type'] == 'T':
                    t_m += 1
            elif cust in self.C:
                d_c += 1
                if self.G.nodes[prov]['type'] == 'T':
                    t_c += 1
            elif cust in self.CP:
                d_cp += 1
                if self.G.nodes[prov]['type'] == 'T':
                    t_cp += 1
            else:
                raise ValueError('Inconsistent data in the graph edge attributes')
        elif e['type'] == 'peer':
            if self.G.nodes[i]['type'] == 'M' and self.G.nodes[j]['type'] == 'M':
                p_m_m += 1
            if self.G.nodes[i]['type'] == 'CP' and self.G.nodes[j]['type'] == 'CP':
                p_cp_cp += 1
            if self.G.nodes[i]['type'] == 'M' and self.G.nodes[j]['type'] == 'CP' or (self.G.nodes[i]['type'] == 'CP' and self.G.nodes[j]['type'] == 'M'):
                p_cp_m += 1
        else:
            raise ValueError('Unexpected data in the graph edge attributes')
    assert d_m / len(self.M) == approx(2 + 2.5 * self.n / 10000, abs=1.0)
    assert d_cp / len(self.CP) == approx(2 + 1.5 * self.n / 10000, abs=1.0)
    assert d_c / len(self.C) == approx(1 + 5 * self.n / 100000, abs=1.0)
    assert p_m_m / len(self.M) == approx(1 + 2 * self.n / 10000, abs=1.0)
    assert p_cp_m / len(self.CP) == approx(0.2 + 2 * self.n / 10000, abs=1.0)
    assert p_cp_cp / len(self.CP) == approx(0.05 + 2 * self.n / 100000, abs=1.0)
    assert t_m / d_m == approx(0.375, abs=0.1)
    assert t_cp / d_cp == approx(0.375, abs=0.1)
    assert t_c / d_c == approx(0.125, abs=0.1)