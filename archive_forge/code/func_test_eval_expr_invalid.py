import pytest
from joblib._utils import eval_expr
@pytest.mark.parametrize('expr', ["exec('import os')", 'print(1)', 'import os', '1+1; import os', '1^1'])
def test_eval_expr_invalid(expr):
    with pytest.raises(ValueError, match='is not a valid or supported arithmetic'):
        eval_expr(expr)