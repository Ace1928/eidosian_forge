import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def test_inference_data_num_chains(self, predictions_data, chains):
    predictions = predictions_data
    inference_data = from_pyro(predictions=predictions, num_chains=chains)
    nchains = inference_data.predictions.sizes['chain']
    assert nchains == chains