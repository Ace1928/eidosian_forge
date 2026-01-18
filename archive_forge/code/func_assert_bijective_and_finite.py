import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform as uniform_lib
def assert_bijective_and_finite(bijector, x, y, event_ndims, atol=0, rtol=1e-05, sess=None):
    """Assert that forward/inverse (along with jacobians) are inverses and finite.

  It is recommended to use x and y values that are very very close to the edge
  of the Bijector's domain.

  Args:
    bijector:  A Bijector instance.
    x:  np.array of values in the domain of bijector.forward.
    y:  np.array of values in the domain of bijector.inverse.
    event_ndims: Integer describing the number of event dimensions this bijector
      operates on.
    atol:  Absolute tolerance.
    rtol:  Relative tolerance.
    sess:  TensorFlow session.  Defaults to the default session.

  Raises:
    AssertionError:  If tests fail.
  """
    sess = sess or ops.get_default_session()
    assert_finite(x)
    assert_finite(y)
    f_x = bijector.forward(x)
    g_y = bijector.inverse(y)
    [x_from_x, y_from_y, ildj_f_x, fldj_x, ildj_y, fldj_g_y, f_x_v, g_y_v] = sess.run([bijector.inverse(f_x), bijector.forward(g_y), bijector.inverse_log_det_jacobian(f_x, event_ndims=event_ndims), bijector.forward_log_det_jacobian(x, event_ndims=event_ndims), bijector.inverse_log_det_jacobian(y, event_ndims=event_ndims), bijector.forward_log_det_jacobian(g_y, event_ndims=event_ndims), f_x, g_y])
    assert_finite(x_from_x)
    assert_finite(y_from_y)
    assert_finite(ildj_f_x)
    assert_finite(fldj_x)
    assert_finite(ildj_y)
    assert_finite(fldj_g_y)
    assert_finite(f_x_v)
    assert_finite(g_y_v)
    np.testing.assert_allclose(x_from_x, x, atol=atol, rtol=rtol)
    np.testing.assert_allclose(y_from_y, y, atol=atol, rtol=rtol)
    np.testing.assert_allclose(-ildj_f_x, fldj_x, atol=atol, rtol=rtol)
    np.testing.assert_allclose(-ildj_y, fldj_g_y, atol=atol, rtol=rtol)