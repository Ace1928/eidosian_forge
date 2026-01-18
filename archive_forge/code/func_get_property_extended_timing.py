import textwrap
import timeit
def get_property_extended_timing(property_args, cached_property):
    """ Time the cases described by the (cached) property depending on an
    extended trait scenario. Wether or not the property is cached is determined
    by the cached_property argument, and the given property_args argument is
    used for the Property trait defintion.

    Parameters
    ----------
    property_args - Str
        The string defining the argument to be passed in the definition of the
        Property trait.  e.g. "depends_on='child.name'"
    cached_property - Str
        The string that will be used to decorate the getter method of the
        Property.  Expected to be either '' or '@cached_property'.

    Returns
    -------
    tuple
        A 4-tuple containing the time to construct the HasTraits subclass, the
        time to instantiate it, the time to reassign child, and the time to
        reassign child.name
    """
    construct_parent_with_property = PARENT_WITH_PROPERTY_CONSTRUCTION_TEMPLATE.format(property_args, cached_property)
    construction_time = timeit.timeit(stmt=construct_parent_with_property, setup=CONSTRUCT_PARENT_SETUP, number=N)
    instantiation_time = timeit.timeit(stmt=INSTANTIATE_PARENT, setup=CONSTRUCT_PARENT_SETUP + construct_parent_with_property, number=N)
    reassign_child_time = timeit.timeit(stmt=REASSIGN_CHILD, setup=CONSTRUCT_PARENT_SETUP + construct_parent_with_property + INSTANTIATE_PARENT, number=N)
    reassign_child_name_time = timeit.timeit(stmt=REASSIGN_CHILD_NAME, setup=CONSTRUCT_PARENT_SETUP + construct_parent_with_property + INSTANTIATE_PARENT, number=N)
    return (construction_time, instantiation_time, reassign_child_time, reassign_child_name_time)