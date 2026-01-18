from typing import cast
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_mxnet
from thinc.types import Array1d, Array2d, IntsXd
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_mxnet, reason='needs MXNet')
def test_mxnet_wrapper_thinc_model_subclass(mx_model):

    class CustomModel(Model):

        def fn(self) -> int:
            return 1337
    model = MXNetWrapper(mx_model, model_class=CustomModel)
    assert isinstance(model, CustomModel)
    assert model.fn() == 1337