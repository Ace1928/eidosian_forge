from collections import defaultdict
def check_once():
    nodes, edges, expected = case13()
    G = Digraph()
    for node in edges:
        G.node(node, shape='rect', label=f'{node}\n' + '\\l'.join(nodes[node]))
    for node, children in edges.items():
        for child in children:
            G.edge(node, child)
    G.view()
    algo = FanoutAlgorithm(nodes, edges, verbose=True)
    got = algo.run()
    assert expected == got