import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_bool_array_input():
    shape = (200, 200)
    center_x, center_y = (140, 150)
    radius = 20
    x, y = np.meshgrid(range(shape[1]), range(shape[0]))
    mask = (y - center_y) ** 2 + (x - center_x) ** 2 < radius ** 2
    src = (center_y, center_x)
    phi = 4 * np.pi / 9.0
    dy = 31 * np.cos(phi)
    dx = 31 * np.sin(phi)
    dst = (center_y + dy, center_x + dx)
    profile_u8 = profile_line(mask.astype(np.uint8), src, dst, mode='reflect')
    assert all(profile_u8[:radius] == 1)
    profile_b = profile_line(mask, src, dst, mode='reflect')
    assert all(profile_b[:radius] == 1)
    assert all(profile_b == profile_u8)