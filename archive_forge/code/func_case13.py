from collections import defaultdict
def case13():
    nodes, edges, _ = case8()
    edges['C'].append('D')
    edges['D'] = ['B']
    expected = {'A': None}
    return (nodes, edges, expected)