from enum import Enum
@staticmethod
def compress_string_factors(string_factors):
    factors = FactorUtils.extract_factors(string_factors)
    compressed_string_factors = FactorUtils.factors_to_string(factors)
    return compressed_string_factors