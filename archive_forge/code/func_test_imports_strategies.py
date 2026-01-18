import textwrap
import pytest
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import assert_run_python_script_without_output
@pytest.mark.xfail(_IS_WASM, reason='cannot start subprocess')
def test_imports_strategies():
    pattern = 'Halving(Grid|Random)SearchCV is experimental'
    good_import = '\n    from sklearn.experimental import enable_halving_search_cv\n    from sklearn.model_selection import HalvingGridSearchCV\n    from sklearn.model_selection import HalvingRandomSearchCV\n    '
    assert_run_python_script_without_output(textwrap.dedent(good_import), pattern=pattern)
    good_import_with_model_selection_first = '\n    import sklearn.model_selection\n    from sklearn.experimental import enable_halving_search_cv\n    from sklearn.model_selection import HalvingGridSearchCV\n    from sklearn.model_selection import HalvingRandomSearchCV\n    '
    assert_run_python_script_without_output(textwrap.dedent(good_import_with_model_selection_first), pattern=pattern)
    bad_imports = f'\n    import pytest\n\n    with pytest.raises(ImportError, match={pattern!r}):\n        from sklearn.model_selection import HalvingGridSearchCV\n\n    import sklearn.experimental\n    with pytest.raises(ImportError, match={pattern!r}):\n        from sklearn.model_selection import HalvingRandomSearchCV\n    '
    assert_run_python_script_without_output(textwrap.dedent(bad_imports), pattern=pattern)