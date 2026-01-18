import tree
def packed_nest_with_indices(structure, flat, index, is_nested_fn, sequence_fn=None):
    """Helper function for pack_sequence_as.

    Args:
        structure: structure to mimic.
        flat: Flattened values to output substructure for.
        index: Index at which to start reading from flat.
        is_nested_fn: Function used to test if a value should
            be treated as a nested structure.
        sequence_fn: Function used to generate a new strcuture instance.

    Returns:
        The tuple (new_index, child), where:
        * new_index - the updated index into `flat`
            having processed `structure`.
        * packed - the subset of `flat` corresponding to `structure`,
            having started at `index`, and packed into the same nested
            format.
    """
    packed = []
    sequence_fn = sequence_fn or tree._sequence_like
    for s in yield_value(structure):
        if is_nested_fn(s):
            new_index, child = packed_nest_with_indices(s, flat, index, is_nested_fn, sequence_fn)
            packed.append(sequence_fn(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return (index, packed)