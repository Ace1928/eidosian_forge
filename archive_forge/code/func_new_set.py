from reportlab.lib import colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
def new_set(self, type='feature', **args):
    """Create a new FeatureSet or GraphSet object.

        Create a new FeatureSet or GraphSet object, add it to the
        track, and return for user manipulation
        """
    type_dict = {'feature': FeatureSet, 'graph': GraphSet}
    set = type_dict[type]()
    for key in args:
        setattr(set, key, args[key])
    set.id = self._next_id
    set.parent = self
    self._sets[self._next_id] = set
    self._next_id += 1
    return set