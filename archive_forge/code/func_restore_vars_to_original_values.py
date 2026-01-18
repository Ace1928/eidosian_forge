def restore_vars_to_original_values(solver, nlp_util_block, mip_util_block):
    """Perform initialization of the subproblem.

    This just restores the continuous variables to the original
    model values, which were saved on the subproblem's utility block when it
    was created.
    """
    _restore_vars_from_nlp_block_saved_values(nlp_util_block)