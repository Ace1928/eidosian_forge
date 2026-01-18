from collections import defaultdict
def case5():
    nodes, edges, _ = case1()
    edges['B'].append('I')
    expected = {'D': None}
    return (nodes, edges, expected)