import pytest
from pandas.errors import (
import pandas as pd
@pytest.mark.parametrize('exc', ['AttributeConflictWarning', 'CSSWarning', 'CategoricalConversionWarning', 'ClosedFileError', 'DataError', 'DatabaseError', 'DtypeWarning', 'EmptyDataError', 'IncompatibilityWarning', 'IndexingError', 'InvalidColumnName', 'InvalidComparison', 'InvalidVersion', 'LossySetitemError', 'MergeError', 'NoBufferPresent', 'NumExprClobberingError', 'NumbaUtilError', 'OptionError', 'OutOfBoundsDatetime', 'ParserError', 'ParserWarning', 'PerformanceWarning', 'PossibleDataLossError', 'PossiblePrecisionLoss', 'PyperclipException', 'SettingWithCopyError', 'SettingWithCopyWarning', 'SpecificationError', 'UnsortedIndexError', 'UnsupportedFunctionCall', 'ValueLabelTypeMismatch'])
def test_exception_importable(exc):
    from pandas import errors
    err = getattr(errors, exc)
    assert err is not None
    msg = '^$'
    with pytest.raises(err, match=msg):
        raise err()