from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim
def get_fill_color(self, tree, node_id):
    if 'rgb' not in self.colors:
        self.colors['rgb'] = _color_brew(tree.n_classes[0])
        if tree.n_outputs != 1:
            self.colors['bounds'] = (np.min(-tree.impurity), np.max(-tree.impurity))
        elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
            self.colors['bounds'] = (np.min(tree.value), np.max(tree.value))
    if tree.n_outputs == 1:
        node_val = tree.value[node_id][0, :]
        if tree.n_classes[0] == 1 and isinstance(node_val, Iterable) and (self.colors['bounds'] is not None):
            node_val = node_val.item()
    else:
        node_val = -tree.impurity[node_id]
    return self.get_color(node_val)