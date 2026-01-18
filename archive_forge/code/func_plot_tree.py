from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim
@validate_params({'decision_tree': [DecisionTreeClassifier, DecisionTreeRegressor], 'max_depth': [Interval(Integral, 0, None, closed='left'), None], 'feature_names': ['array-like', None], 'class_names': ['array-like', 'boolean', None], 'label': [StrOptions({'all', 'root', 'none'})], 'filled': ['boolean'], 'impurity': ['boolean'], 'node_ids': ['boolean'], 'proportion': ['boolean'], 'rounded': ['boolean'], 'precision': [Interval(Integral, 0, None, closed='left'), None], 'ax': 'no_validation', 'fontsize': [Interval(Integral, 0, None, closed='left'), None]}, prefer_skip_nested_validation=True)
def plot_tree(decision_tree, *, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, impurity=True, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None):
    """Plot a decision tree.

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    The visualization is fit automatically to the size of the axis.
    Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
    the size of the rendering.

    Read more in the :ref:`User Guide <tree>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    decision_tree : decision tree regressor or classifier
        The decision tree to be plotted.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : array-like of str, default=None
        Names of each of the features.
        If None, generic names will be used ("x[0]", "x[1]", ...).

    class_names : array-like of str or True, default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, default=False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    impurity : bool, default=True
        When set to ``True``, show the impurity at each node.

    node_ids : bool, default=False
        When set to ``True``, show the ID number on each node.

    proportion : bool, default=False
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    ax : matplotlib axis, default=None
        Axes to plot to. If None, use current axis. Any previous content
        is cleared.

    fontsize : int, default=None
        Size of text font. If None, determined automatically to fit figure.

    Returns
    -------
    annotations : list of artists
        List containing the artists for the annotation boxes making up the
        tree.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.plot_tree(clf)
    [...]
    """
    check_is_fitted(decision_tree)
    exporter = _MPLTreeExporter(max_depth=max_depth, feature_names=feature_names, class_names=class_names, label=label, filled=filled, impurity=impurity, node_ids=node_ids, proportion=proportion, rounded=rounded, precision=precision, fontsize=fontsize)
    return exporter.export(decision_tree, ax=ax)