import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def close_or_save(pdf, fig):
    if pdf_output:
        pdf.savefig(fig)