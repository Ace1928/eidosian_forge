import triton
import triton.language as tl
@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and (not b_isnan)
        equal |= a_isnan and b_isnan
    mask |= equal & (a_index < b_index)
    return (tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index))