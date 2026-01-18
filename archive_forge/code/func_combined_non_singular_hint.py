def combined_non_singular_hint(operator_a, operator_b):
    """Get combined hint for when ."""
    if operator_a.is_non_singular is False or operator_b.is_non_singular is False:
        return False
    return operator_a.is_non_singular and operator_b.is_non_singular