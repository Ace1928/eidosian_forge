from io import BytesIO
from ... import errors, tests, ui
from . import TestCaseWithBranch
def make_refs(self, branch):
    needed_refs = branch._get_check_refs()
    refs = {}
    distances = set()
    existences = set()
    for ref in needed_refs:
        kind, value = ref
        if kind == 'lefthand-distance':
            distances.add(value)
        elif kind == 'revision-existence':
            existences.add(value)
        else:
            raise AssertionError('unknown ref kind for ref %s' % ref)
    node_distances = branch.repository.get_graph().find_lefthand_distances(distances)
    for key, distance in node_distances.items():
        refs['lefthand-distance', key] = distance
        if key in existences and distance > 0:
            refs['revision-existence', key] = True
            existences.remove(key)
    parent_map = branch.repository.get_graph().get_parent_map(existences)
    for key in parent_map:
        refs['revision-existence', key] = True
        existences.remove(key)
    for key in existences:
        refs['revision-existence', key] = False
    return refs