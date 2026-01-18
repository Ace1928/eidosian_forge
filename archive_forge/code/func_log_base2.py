from math import log
def log_base2(score):
    """Convenience function for computing logarithms with base 2."""
    if score == 0.0:
        return NEG_INF
    return log(score, 2)