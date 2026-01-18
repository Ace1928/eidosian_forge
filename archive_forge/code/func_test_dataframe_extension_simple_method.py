import modin.pandas as pd
from modin.pandas.api.extensions import register_dataframe_accessor
def test_dataframe_extension_simple_method():
    expected_string_val = 'Some string value'
    method_name = 'new_method'
    df = pd.DataFrame([1, 2, 3])

    @register_dataframe_accessor(method_name)
    def my_method_implementation(self):
        return expected_string_val
    assert method_name in pd.dataframe._DATAFRAME_EXTENSIONS_.keys()
    assert pd.dataframe._DATAFRAME_EXTENSIONS_[method_name] is my_method_implementation
    assert df.new_method() == expected_string_val