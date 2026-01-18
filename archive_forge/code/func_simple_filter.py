def simple_filter(data=None, attr=None, value=None, property_field=None):
    """Filter a list of dicts

    :param list data:
        The list to be filtered.  The list is modified in-place and will
        be changed if any filtering occurs.
    :param string attr:
        The name of the attribute to filter.  If attr does not exist no
        match will succeed and no rows will be returned.  If attr is
        None no filtering will be performed and all rows will be returned.
    :param string value:
        The value to filter.  None is considered to be a 'no filter' value.
        '' matches against a Python empty string.
    :param string property_field:
        The name of the data field containing a property dict to filter.
        If property_field is None, attr is a field name. If property_field
        is not None, attr is a property key name inside the named property
        field.

    :returns:
        Returns the filtered list
    :rtype list:

    This simple filter (one attribute, one exact-match value) searches a
    list of dicts to select items.  It first searches the item dict for a
    matching ``attr`` then does an exact-match on the ``value``.  If
    ``property_field`` is given, it will look inside that field (if it
    exists and is a dict) for a matching ``value``.
    """
    if not data or not attr or value is None:
        return data
    for d in reversed(data):
        if attr in d:
            search_value = d[attr]
        elif property_field and property_field in d and isinstance(d[property_field], dict):
            if attr in d[property_field]:
                search_value = d[property_field][attr]
            else:
                search_value = None
        else:
            search_value = None
        if not search_value or search_value != value:
            try:
                data.remove(d)
            except ValueError:
                pass
    return data