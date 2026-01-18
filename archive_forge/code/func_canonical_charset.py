def canonical_charset(charset):
    """Returns the canonical or preferred name of a charset.

    Additional character sets can be recognized by this function by
    altering the character_set_aliases dictionary in this module.
    Charsets which are not recognized are simply converted to
    upper-case (as charset names are always case-insensitive).
    
    See <http://www.iana.org/assignments/character-sets>.

    """
    if not charset:
        return charset
    uc = charset.upper()
    uccon = character_set_aliases.get(uc, uc)
    return uccon