def replace_entry(line, fieldn, newentry):
    """Replace an entry in a string by the field number.

    No padding is implemented currently.  Spacing will change if
    the original field entry and the new field entry are of
    different lengths.
    """
    start = _find_start_entry(line, fieldn)
    leng = len(line[start:].split()[0])
    newline = line[:start] + str(newentry) + line[start + leng:]
    return newline