from ..pari import pari
import fractions
def simultaneous_smith_normal_form(in1, in2):
    u1, v1, d1 = _smith_normal_form_with_inverse(in1)
    u2, v2, d2 = _bottom_row_stable_smith_normal_form(matrix_mult(matrix_inverse(v1), in2))
    assert _change_coordinates(u2, v2, matrix_mult(matrix_inverse(v1), in2)) == d2
    return (u1, matrix_mult(v1, u2), v2, d1, d2)