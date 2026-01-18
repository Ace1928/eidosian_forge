import textwrap
import timeit
def get_property_simple_timing(property_args, cached_property):
    """ Time the cases described by the (cached) property depending on a simple
    trait scenario.  Whether or not the property is cached is based on the
    cached_property argument, and the given property_args argument is used in
    the Property trait defintion.

    Parameters
    ----------
    property_args - Str
        The string defining the argument to be passed in the definition of the
        Property trait.  e.g. "depends_on='name'"
    cached_property - Str
        The string that will be used to decorate the getter method of the
        Property.  Expected to be either '' or '@cached_property'.

    Returns
    -------
    tuple
        A 3-tuple containing the time to construct the HasTraits subclass, the
        time to instantiate it, and the time to reassign the trait being
        depended-on / observed.
    """
    construct_person_with_property = PERSON_WITH_PROPERTY_CONSTRUCTION_TEMPLATE.format(property_args, cached_property)
    construction_time = timeit.timeit(stmt=construct_person_with_property, setup=BASE_SETUP, number=N)
    instantiation_time = timeit.timeit(stmt=INSTANTIATE_PERSON, setup=BASE_SETUP + construct_person_with_property, number=N)
    reassign_dependee_name_time = timeit.timeit(stmt=REASSIGN_NAME, setup=BASE_SETUP + construct_person_with_property + INSTANTIATE_PERSON, number=N)
    return (construction_time, instantiation_time, reassign_dependee_name_time)