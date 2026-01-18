from ..sage_helper import _within_sage, sage_method
from .cuspCrossSection import RealCuspCrossSection
from .squareExtensions import find_shapes_as_complex_sqrt_lin_combinations
from . import verifyHyperbolicity
from . import exceptions
from ..exceptions import SnapPeaFatalError
from ..snap import t3mlite as t3m
def get_opacity(tilt):
    sign, interval = tilt.sign_with_interval()
    if sign < 0:
        return True
    if sign == 0:
        return False
    if sign > 0:
        raise exceptions.TiltProvenPositiveNumericalVerifyError(interval)