import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_function_transformer_with_dataframe_and_check_inverse_True():
    """Check error is raised when check_inverse=True.

    Non-regresion test for gh-25261.
    """
    pd = pytest.importorskip('pandas')
    transformer = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x, check_inverse=True)
    df_mixed = pd.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
    msg = "'check_inverse' is only supported when all the elements in `X` is numerical."
    with pytest.raises(ValueError, match=msg):
        transformer.fit(df_mixed)