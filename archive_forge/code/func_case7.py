from collections import defaultdict
def case7():
    nodes, edges, _ = case1()
    edges['I'].append('M')
    expected = {'D': None}
    return (nodes, edges, expected)