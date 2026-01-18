import modin.pandas as pd
from modin.pandas.api.extensions import register_dataframe_accessor
def test_dataframe_extension_accessing_existing_methods():
    df = pd.DataFrame([1, 2, 3])
    method_name = 'self_accessor'
    expected_result = df.sum() / df.count()

    @register_dataframe_accessor(method_name)
    def my_average(self):
        return self.sum() / self.count()
    assert method_name in pd.dataframe._DATAFRAME_EXTENSIONS_.keys()
    assert pd.dataframe._DATAFRAME_EXTENSIONS_[method_name] is my_average
    assert df.self_accessor().equals(expected_result)