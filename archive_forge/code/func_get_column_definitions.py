import operator
def get_column_definitions(attr_map, long_listing):
    """Return table headers and column names for a listing table.

    An attribute map (attr_map) is a list of table entry definitions
    and the format of the map is as follows:

    :param attr_map: a list of table entry definitions.
        Each entry should be a tuple consisting of
        (API attribute name, header name, listing mode). For example:

        .. code-block:: python

           (('id', 'ID', LIST_BOTH),
            ('name', 'Name', LIST_BOTH),
            ('tenant_id', 'Project', LIST_LONG_ONLY))

        The third field of each tuple must be one of LIST_BOTH,
        LIST_LONG_ONLY (a corresponding column is shown only in a long mode),
        or LIST_SHORT_ONLY (a corresponding column is shown only
        in a short mode).
    :param long_listing: A boolean value which indicates a long listing
        or not. In most cases, parsed_args.long is passed to this argument.
    :return: A tuple of a list of table headers and a list of column names.

    """
    if long_listing:
        headers = [hdr for col, hdr, listing_mode in attr_map if listing_mode in (LIST_BOTH, LIST_LONG_ONLY)]
        columns = [col for col, hdr, listing_mode in attr_map if listing_mode in (LIST_BOTH, LIST_LONG_ONLY)]
    else:
        headers = [hdr for col, hdr, listing_mode in attr_map if listing_mode if listing_mode in (LIST_BOTH, LIST_SHORT_ONLY)]
        columns = [col for col, hdr, listing_mode in attr_map if listing_mode if listing_mode in (LIST_BOTH, LIST_SHORT_ONLY)]
    return (headers, columns)