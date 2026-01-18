def needs_update(targ_capacity, curr_capacity, num_up_to_date):
    """Return whether there are more batch updates to do.

    Inputs are the target size for the group, the current size of the group,
    and the number of members that already have the latest definition.
    """
    return not num_up_to_date >= curr_capacity == targ_capacity