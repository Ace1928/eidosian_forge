import math
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .basic import Booster, _data_from_pandas, _is_zero, _log_warning, _MissingType
from .compat import GRAPHVIZ_INSTALLED, MATPLOTLIB_INSTALLED, pd_DataFrame
from .sklearn import LGBMModel
def create_tree_digraph(booster: Union[Booster, LGBMModel], tree_index: int=0, show_info: Optional[List[str]]=None, precision: Optional[int]=3, orientation: str='horizontal', example_case: Optional[Union[np.ndarray, pd_DataFrame]]=None, max_category_values: int=10, **kwargs: Any) -> Any:
    """Create a digraph representation of specified tree.

    Each node in the graph represents a node in the tree.

    Non-leaf nodes have labels like ``Column_10 <= 875.9``, which means
    "this node splits on the feature named "Column_10", with threshold 875.9".

    Leaf nodes have labels like ``leaf 2: 0.422``, which means "this node is a
    leaf node, and the predicted value for records that fall into this node
    is 0.422". The number (``2``) is an internal unique identifier and doesn't
    have any special meaning.

    .. note::

        For more information please visit
        https://graphviz.readthedocs.io/en/stable/api.html#digraph.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance to be converted.
    tree_index : int, optional (default=0)
        The index of a target tree to convert.
    show_info : list of str, or None, optional (default=None)
        What information should be shown in nodes.

            - ``'split_gain'`` : gain from adding this split to the model
            - ``'internal_value'`` : raw predicted value that would be produced by this node if it was a leaf node
            - ``'internal_count'`` : number of records from the training data that fall into this non-leaf node
            - ``'internal_weight'`` : total weight of all nodes that fall into this non-leaf node
            - ``'leaf_count'`` : number of records from the training data that fall into this leaf node
            - ``'leaf_weight'`` : total weight (sum of Hessian) of all observations that fall into this leaf node
            - ``'data_percentage'`` : percentage of training data that fall into this node
    precision : int or None, optional (default=3)
        Used to restrict the display of floating point values to a certain precision.
    orientation : str, optional (default='horizontal')
        Orientation of the tree.
        Can be 'horizontal' or 'vertical'.
    example_case : numpy 2-D array, pandas DataFrame or None, optional (default=None)
        Single row with the same structure as the training data.
        If not None, the plot will highlight the path that sample takes through the tree.

        .. versionadded:: 4.0.0

    max_category_values : int, optional (default=10)
        The maximum number of category values to display in tree nodes, if the number of thresholds is greater than this value, thresholds will be collapsed and displayed on the label tooltip instead.

        .. warning::

            Consider wrapping the SVG string of the tree graph with ``IPython.display.HTML`` when running on JupyterLab to get the `tooltip <https://graphviz.org/docs/attrs/tooltip>`_ working right.

            Example:

            .. code-block:: python

                from IPython.display import HTML

                graph = lgb.create_tree_digraph(clf, max_category_values=5)
                HTML(graph._repr_image_svg_xml())

        .. versionadded:: 4.0.0

    **kwargs
        Other parameters passed to ``Digraph`` constructor.
        Check https://graphviz.readthedocs.io/en/stable/api.html#digraph for the full list of supported parameters.

    Returns
    -------
    graph : graphviz.Digraph
        The digraph representation of specified tree.
    """
    if isinstance(booster, LGBMModel):
        booster = booster.booster_
    elif not isinstance(booster, Booster):
        raise TypeError('booster must be Booster or LGBMModel.')
    model = booster.dump_model()
    tree_infos = model['tree_info']
    feature_names = model.get('feature_names', None)
    monotone_constraints = model.get('monotone_constraints', None)
    if tree_index < len(tree_infos):
        tree_info = tree_infos[tree_index]
    else:
        raise IndexError('tree_index is out of range.')
    if show_info is None:
        show_info = []
    if example_case is not None:
        if not isinstance(example_case, (np.ndarray, pd_DataFrame)) or example_case.ndim != 2:
            raise ValueError('example_case must be a numpy 2-D array or a pandas DataFrame')
        if example_case.shape[0] != 1:
            raise ValueError('example_case must have a single row.')
        if isinstance(example_case, pd_DataFrame):
            example_case = _data_from_pandas(data=example_case, feature_name='auto', categorical_feature='auto', pandas_categorical=booster.pandas_categorical)[0]
        example_case = example_case[0]
    return _to_graphviz(tree_info=tree_info, show_info=show_info, feature_names=feature_names, precision=precision, orientation=orientation, constraints=monotone_constraints, example_case=example_case, max_category_values=max_category_values, **kwargs)