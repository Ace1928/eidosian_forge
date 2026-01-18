import warnings
import pytest
from nibabel import pkg_info
from nibabel.deprecated import (
from nibabel.tests.test_deprecator import TestDeprecatorFunc as _TestDF
def test_alert_future_error():
    with pytest.warns(FutureWarning):
        alert_future_error('Message', '9999.9.9', warning_rec='Silence this warning by doing XYZ.', error_rec='Fix this issue by doing XYZ.')
    with pytest.raises(RuntimeError):
        alert_future_error('Message', '1.0.0', warning_rec='Silence this warning by doing XYZ.', error_rec='Fix this issue by doing XYZ.')
    with pytest.raises(ValueError):
        alert_future_error('Message', '1.0.0', warning_rec='Silence this warning by doing XYZ.', error_rec='Fix this issue by doing XYZ.', error_class=ValueError)
    with pytest.raises(ValueError):
        alert_future_error('Message', '2.0.0', warning_rec='Silence this warning by doing XYZ.', error_rec='Fix this issue by doing XYZ.', error_class=ValueError)