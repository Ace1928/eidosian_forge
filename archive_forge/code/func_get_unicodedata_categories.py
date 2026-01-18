def get_unicodedata_categories():
    """
    Extracts Unicode categories information from unicodedata library. Each category is
    represented with an ordered list containing code points and code point ranges.

    :return: a dictionary with category names as keys and lists as values.
    """
    categories = {k: [] for k in ('C', 'Cc', 'Cf', 'Cs', 'Co', 'Cn', 'L', 'Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'M', 'Mn', 'Mc', 'Me', 'N', 'Nd', 'Nl', 'No', 'P', 'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po', 'S', 'Sm', 'Sc', 'Sk', 'So', 'Z', 'Zs', 'Zl', 'Zp')}
    major_category = 'C'
    start_cp, next_cp = (0, 1)
    for cp in range(maxunicode + 1):
        if category(chr(cp))[0] != major_category:
            if cp > next_cp:
                categories[major_category].append((start_cp, cp))
            else:
                categories[major_category].append(start_cp)
            major_category = category(chr(cp))[0]
            start_cp, next_cp = (cp, cp + 1)
    else:
        if next_cp == maxunicode + 1:
            categories[major_category].append(start_cp)
        else:
            categories[major_category].append((start_cp, maxunicode + 1))
    minor_category = 'Cc'
    start_cp, next_cp = (0, 1)
    for cp in range(maxunicode + 1):
        if category(chr(cp)) != minor_category:
            if cp > next_cp:
                categories[minor_category].append((start_cp, cp))
            else:
                categories[minor_category].append(start_cp)
            minor_category = category(chr(cp))
            start_cp, next_cp = (cp, cp + 1)
    else:
        if next_cp == maxunicode + 1:
            categories[minor_category].append(start_cp)
        else:
            categories[minor_category].append((start_cp, maxunicode + 1))
    return categories