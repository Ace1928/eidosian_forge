import textwrap
import timeit
def get_decorated_method_extended_timing(decorator):
    """ Time the cases described by the decorated method with enxtended trait
    case, using the given method decorator.

    Parameters
    ----------
    decorator - Str
        The string defining the decorator to be used on the method of the
        HasTraits subclass.  e.g. "@observe('child.name')"

    Returns
    -------
    tuple
        A 4-tuple containing the time to construct the HasTraits subclass, the
        time to instantiate it, the time to reassign child, and the
        time to reassign child.name
    """
    construct_parent = PARENT_CONSTRUCTION_TEMPLATE.format(decorator=decorator)
    construction_time = timeit.timeit(stmt=construct_parent, setup=CONSTRUCT_PARENT_SETUP, number=N)
    instantiation_time = timeit.timeit(stmt=INSTANTIATE_PARENT, setup=CONSTRUCT_PARENT_SETUP + construct_parent, number=N)
    reassign_child_time = timeit.timeit(stmt=REASSIGN_CHILD, setup=CONSTRUCT_PARENT_SETUP + construct_parent + INSTANTIATE_PARENT, number=N)
    reassign_child_name_time = timeit.timeit(stmt=REASSIGN_CHILD_NAME, setup=CONSTRUCT_PARENT_SETUP + construct_parent + INSTANTIATE_PARENT, number=N)
    return (construction_time, instantiation_time, reassign_child_time, reassign_child_name_time)