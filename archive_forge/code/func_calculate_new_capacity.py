import math
def calculate_new_capacity(current, adjustment, adjustment_type, min_adjustment_step, minimum, maximum):
    """Calculates new capacity from the given adjustments.

    Given the current capacity, calculates the new capacity which results
    from applying the given adjustment of the given adjustment-type.  The
    new capacity will be kept within the maximum and minimum bounds.
    """

    def _get_minimum_adjustment(adjustment, min_adjustment_step):
        if min_adjustment_step and min_adjustment_step > abs(adjustment):
            adjustment = min_adjustment_step if adjustment > 0 else -min_adjustment_step
        return adjustment
    if adjustment_type in (CHANGE_IN_CAPACITY, CFN_CHANGE_IN_CAPACITY):
        new_capacity = current + adjustment
    elif adjustment_type in (EXACT_CAPACITY, CFN_EXACT_CAPACITY):
        new_capacity = adjustment
    else:
        delta = current * adjustment / 100.0
        if math.fabs(delta) < 1.0:
            rounded = int(math.ceil(delta) if delta > 0.0 else math.floor(delta))
        else:
            rounded = int(math.floor(delta) if delta > 0.0 else math.ceil(delta))
        adjustment = _get_minimum_adjustment(rounded, min_adjustment_step)
        new_capacity = current + adjustment
    if new_capacity > maximum:
        return maximum
    if new_capacity < minimum:
        return minimum
    return new_capacity