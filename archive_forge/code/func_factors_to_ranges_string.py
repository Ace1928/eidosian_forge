from enum import Enum
@staticmethod
def factors_to_ranges_string(factors_set):
    if factors_set is None or len(factors_set) == 0:
        return 'None'
    grouped_factors = FactorUtils.group_factors_by_range(factors_set)
    return ';'.join([FactorUtils.single_range_to_string(min(x), max(x)) for x in grouped_factors])