import modin.pandas as pd
from modin.pandas.api.extensions import register_dataframe_accessor
def test_dataframe_extension_non_method():
    expected_val = 4
    attribute_name = 'four'
    register_dataframe_accessor(attribute_name)(expected_val)
    df = pd.DataFrame([1, 2, 3])
    assert attribute_name in pd.dataframe._DATAFRAME_EXTENSIONS_.keys()
    assert pd.dataframe._DATAFRAME_EXTENSIONS_[attribute_name] == 4
    assert df.four == expected_val