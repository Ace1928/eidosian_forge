from math import cos, exp, sin, sqrt, atan2
def dist_cie2000(rgb1, rgb2):
    """
    Determine distance between two rgb colors using the CIE2000 algorithm.

    :arg tuple rgb1: RGB color definition
    :arg tuple rgb2: RGB color definition
    :returns: Square of the distance between provided colors
    :rtype: float

    For efficiency, the square of the distance is returned
    which is sufficient for comparisons
    """
    s_l = k_l = k_c = k_h = 1
    l_1, a_1, b_1 = rgb_to_lab(*rgb1)
    l_2, a_2, b_2 = rgb_to_lab(*rgb2)
    delta_l = l_2 - l_1
    l_mean = (l_1 + l_2) / 2
    c_1 = sqrt(a_1 ** 2 + b_1 ** 2)
    c_2 = sqrt(a_2 ** 2 + b_2 ** 2)
    c_mean = (c_1 + c_2) / 2
    delta_c = c_1 - c_2
    g_x = sqrt(c_mean ** 7 / (c_mean ** 7 + 25 ** 7))
    h_1 = atan2(b_1, a_1 + a_1 / 2 * (1 - g_x)) % 360
    h_2 = atan2(b_2, a_2 + a_2 / 2 * (1 - g_x)) % 360
    if 0 in (c_1, c_2):
        delta_h_prime = 0
        h_mean = h_1 + h_2
    else:
        delta_h_prime = h_2 - h_1
        if abs(delta_h_prime) <= 180:
            h_mean = (h_1 + h_2) / 2
        else:
            if h_2 <= h_1:
                delta_h_prime += 360
            else:
                delta_h_prime -= 360
            h_mean = (h_1 + h_2 + 360) / 2 if h_1 + h_2 < 360 else (h_1 + h_2 - 360) / 2
    delta_h = 2 * sqrt(c_1 * c_2) * sin(delta_h_prime / 2)
    t_x = 1 - 0.17 * cos(h_mean - 30) + 0.24 * cos(2 * h_mean) + 0.32 * cos(3 * h_mean + 6) - 0.2 * cos(4 * h_mean - 63)
    s_l = 1 + 0.015 * (l_mean - 50) ** 2 / sqrt(20 + (l_mean - 50) ** 2)
    s_c = 1 + 0.045 * c_mean
    s_h = 1 + 0.015 * c_mean * t_x
    r_t = -2 * g_x * sin(abs(60 * exp(-1 * abs((delta_h - 275) / 25) ** 2)))
    delta_l = delta_l / (k_l * s_l)
    delta_c = delta_c / (k_c * s_c)
    delta_h = delta_h / (k_h * s_h)
    return delta_l ** 2 + delta_c ** 2 + delta_h ** 2 + r_t * delta_c * delta_h