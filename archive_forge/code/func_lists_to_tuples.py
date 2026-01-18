import tree
def lists_to_tuples(structure):

    def sequence_fn(instance, args):
        if isinstance(instance, list):
            return tuple(args)
        return tree._sequence_like(instance, args)
    return pack_sequence_as(structure, tree.flatten(structure), sequence_fn=sequence_fn)