from collections import defaultdict
def case10():
    nodes, edges, _ = case8()
    edges['C'].append('A')
    expected = {'A': {'C'}}
    return (nodes, edges, expected)