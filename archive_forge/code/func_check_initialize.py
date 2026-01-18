import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def check_initialize(model, inputs):
    model.initialize()
    model.initialize(X=inputs)
    model.initialize(X=inputs, Y=model.predict(inputs))