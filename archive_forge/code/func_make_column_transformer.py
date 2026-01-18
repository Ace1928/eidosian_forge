import warnings
from collections import Counter
from itertools import chain
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from ..base import TransformerMixin, _fit_context, clone
from ..pipeline import _fit_transform_one, _name_estimators, _transform_one
from ..preprocessing import FunctionTransformer
from ..utils import Bunch, _get_column_indices, _safe_indexing
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._metadata_requests import METHODS
from ..utils._param_validation import HasMethods, Hidden, Interval, StrOptions
from ..utils._set_output import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _BaseComposition
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
def make_column_transformer(*transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None, verbose=False, verbose_feature_names_out=True):
    """Construct a ColumnTransformer from the given transformers.

    This is a shorthand for the ColumnTransformer constructor; it does not
    require, and does not permit, naming the transformers. Instead, they will
    be given names automatically based on their types. It also does not allow
    weighting with ``transformer_weights``.

    Read more in the :ref:`User Guide <make_column_transformer>`.

    Parameters
    ----------
    *transformers : tuples
        Tuples of the form (transformer, columns) specifying the
        transformer objects to be applied to subsets of the data.

        transformer : {'drop', 'passthrough'} or estimator
            Estimator must support :term:`fit` and :term:`transform`.
            Special-cased strings 'drop' and 'passthrough' are accepted as
            well, to indicate to drop the columns or to pass them through
            untransformed, respectively.
        columns : str,  array-like of str, int, array-like of int, slice,                 array-like of bool or callable
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name. A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.
            A callable is passed the input data `X` and can return any of the
            above. To select multiple columns by name or dtype, you can use
            :obj:`make_column_selector`.

    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers` will be automatically passed
        through. This subset of columns is concatenated with the output of
        the transformers.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support :term:`fit` and :term:`transform`.

    sparse_threshold : float, default=0.3
        If the transformed output consists of a mix of sparse and dense data,
        it will be stacked as a sparse matrix if the density is lower than this
        value. Use ``sparse_threshold=0`` to always return dense.
        When the transformed output consists of all sparse or all dense data,
        the stacked result will be sparse or dense, respectively, and this
        keyword will be ignored.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    verbose_feature_names_out : bool, default=True
        If True, :meth:`ColumnTransformer.get_feature_names_out` will prefix
        all feature names with the name of the transformer that generated that
        feature.
        If False, :meth:`ColumnTransformer.get_feature_names_out` will not
        prefix any feature names and will error if feature names are not
        unique.

        .. versionadded:: 1.0

    Returns
    -------
    ct : ColumnTransformer
        Returns a :class:`ColumnTransformer` object.

    See Also
    --------
    ColumnTransformer : Class that allows combining the
        outputs of multiple transformer objects used on column subsets
        of the data into a single feature space.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
    >>> from sklearn.compose import make_column_transformer
    >>> make_column_transformer(
    ...     (StandardScaler(), ['numerical_column']),
    ...     (OneHotEncoder(), ['categorical_column']))
    ColumnTransformer(transformers=[('standardscaler', StandardScaler(...),
                                     ['numerical_column']),
                                    ('onehotencoder', OneHotEncoder(...),
                                     ['categorical_column'])])
    """
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(transformer_list, n_jobs=n_jobs, remainder=remainder, sparse_threshold=sparse_threshold, verbose=verbose, verbose_feature_names_out=verbose_feature_names_out)