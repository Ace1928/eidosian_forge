import logging
import os
import numpy as np
import pandas as pd
import pytest
@pytest.fixture()
def reset_randomstate():
    """
    Fixture that set the global RandomState to the fixed seed 1

    Notes
    -----
    Used by passing as an argument to the function that uses the global
    RandomState

    def test_some_plot(reset_randomstate):
        <test code>

    Returns the state after the test function exits
    """
    state = np.random.get_state()
    np.random.seed(1)
    yield
    np.random.set_state(state)