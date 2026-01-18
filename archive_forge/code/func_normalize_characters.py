def normalize_characters(tag):
    """
    BCP 47 is case-insensitive, and CLDR's use of it considers underscores
    equivalent to hyphens. So here we smash tags into lowercase with hyphens,
    so we can make exact comparisons.

    >>> normalize_characters('en_US')
    'en-us'
    >>> normalize_characters('zh-Hant_TW')
    'zh-hant-tw'
    """
    return tag.lower().replace('_', '-')