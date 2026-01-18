import os
import collections.abc
def architecture_is_concerned(self, architecture, architecture_restrictions, allow_mixing_positive_and_negative=False):
    """Determine if a dpkg architecture is part of a list of restrictions [debarch_is_concerned]

        This method is the closest match to dpkg's Dpkg::Arch::debarch_is_concerned function.

        Compatibility notes:
          * The Dpkg::Arch::debarch_is_concerned function allow matching of negative and positive
            restrictions by default.  Often, this behaviour is not allowed nor recommended and the
            Debian Policy does not allow this practice in e.g., Build-Depends.  Therefore, this
            implementation defaults to raising ValueError when this occurs.  If the original
            behaviour is needed, set `allow_mixing_positive_and_negative` to True.
          * The Dpkg::Arch::debarch_is_concerned function is lazy and exits as soon as it finds a
            match. This means that if negative and positive restrictions are mixed, then order of
            the matches are important. This adaption matches that behaviour (provided that
            `allow_mixing_positive_and_negative` is set to True)

        >>> arch_table = DpkgArchTable.load_arch_table()
        >>> arch_table.architecture_is_concerned("linux-amd64", ["amd64", "i386"])
        True
        >>> arch_table.architecture_is_concerned("amd64", ["!amd64", "!i386"])
        False
        >>> # This is False because the "!amd64" is matched first.
        >>> arch_table.architecture_is_concerned("linux-amd64", ["!linux-amd64", "linux-any"],
        ...                                      allow_mixing_positive_and_negative=True)
        False
        >>> # This is True because the "linux-any" is matched first.
        >>> arch_table.architecture_is_concerned("linux-amd64", ["linux-any", "!linux-amd64"],
        ...                                      allow_mixing_positive_and_negative=True)
        True

        :param architecture: A string representing a dpkg architecture/wildcard.
        :param architecture_restrictions: A list of positive (amd64) or negative (!amd64) dpkg
                                          architectures or/and wildcards.
        :param allow_mixing_positive_and_negative: If True, the `architecture_restrictions` list
                                       can mix positive and negative (e.g., ["!any-amd64", "any"])
                                       restrictions. If False, mixing will trigger a ValueError.
        :returns: True if `architecture` is accepted by the `architecture_restrictions`.
        """
    verdict = None
    positive_match_seen = False
    negative_match_seen = False
    arch_restriction_iter = iter(architecture_restrictions)
    try:
        dpkg_arch = self._dpkg_arch_to_tuple(architecture)
    except KeyError:
        return False
    for arch_restriction in arch_restriction_iter:
        if arch_restriction == '':
            continue
        if arch_restriction[0] == '!':
            negative_match_seen = True
        else:
            positive_match_seen = True
        if verdict is not None:
            continue
        arch_restriction = arch_restriction.lower()
        verdict_if_matched = True
        arch_restriction_positive = arch_restriction
        if arch_restriction[0] == '!':
            verdict_if_matched = False
            arch_restriction_positive = arch_restriction[1:]
        dpkg_wildcard = self._dpkg_wildcard_to_tuple(arch_restriction_positive)
        if dpkg_arch in dpkg_wildcard:
            verdict = verdict_if_matched
            if allow_mixing_positive_and_negative:
                return verdict
    if not allow_mixing_positive_and_negative and positive_match_seen and negative_match_seen:
        raise ValueError('architecture_restrictions contained mixed positive and negativerestrictions (and allow_mixing_positive_and_negative was not True)')
    if verdict is None:
        verdict = negative_match_seen
    return verdict