from ._gi import \
def generate_doc_string(info):
    """Generate a doc string given a GIInfoStruct.

    :param gi.types.BaseInfo info:
        GI info instance to generate documentation for.
    :returns:
        Generated documentation as a string.
    :rtype: str

    This passes the info struct to the currently registered doc string
    generator and returns the result.
    """
    return _generate_doc_string_func(info)