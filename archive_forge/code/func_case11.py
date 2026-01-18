from collections import defaultdict
def case11():
    nodes, edges, _ = case8()
    edges['C'].append('D')
    edges['D'] = []
    expected = {'A': {'C'}}
    return (nodes, edges, expected)