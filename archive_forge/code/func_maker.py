import pytest
import pandas as pd
def maker(dct, is_categorical=False):
    df = pd.DataFrame(dct)
    return df.astype('category') if is_categorical else df