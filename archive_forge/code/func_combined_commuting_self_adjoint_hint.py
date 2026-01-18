def combined_commuting_self_adjoint_hint(operator_a, operator_b):
    """Get combined hint for self-adjoint-ness."""
    if operator_a.is_self_adjoint and operator_b.is_self_adjoint:
        return True
    if operator_a.is_self_adjoint is True and operator_b.is_self_adjoint is False or (operator_a.is_self_adjoint is False and operator_b.is_self_adjoint is True):
        return False
    return None