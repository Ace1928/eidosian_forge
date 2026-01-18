import os
import collections.abc
def matches_architecture(self, architecture, alias):
    """Determine if a dpkg architecture matches another architecture or a wildcard [debarch_is]

        This method is the closest match to dpkg's Dpkg::Arch::debarch_is function.

        >>> arch_table = DpkgArchTable.load_arch_table()
        >>> arch_table.matches_architecture("amd64", "linux-any")
        True
        >>> arch_table.matches_architecture("i386", "linux-any")
        True
        >>> arch_table.matches_architecture("amd64", "amd64")
        True
        >>> arch_table.matches_architecture("i386", "amd64")
        False
        >>> arch_table.matches_architecture("all", "amd64")
        False
        >>> arch_table.matches_architecture("all", "all")
        True
        >>> # i386 is the short form of linux-i386. Therefore, it does not match kfreebsd-i386
        >>> arch_table.matches_architecture("i386", "kfreebsd-i386")
        False
        >>> # Note that "armel" and "armhf" are "arm" CPUs, so it is matched by "any-arm"
        >>> # (similar holds for some other architecture <-> CPU name combinations)
        >>> all(arch_table.matches_architecture(n, 'any-arm') for n in ['armel', 'armhf'])
        True
        >>> # Since "armel" is not a valid CPU name, this returns False (the correct would be
        >>> # any-arm as noted above)
        >>> arch_table.matches_architecture("armel", "any-armel")
        False
        >>> # Wildcards used as architecture always fail (except for special cases noted in the
        >>> # compatibility notes below)
        >>> arch_table.matches_architecture("any-i386", "i386")
        False
        >>> # any-i386 is not a subset of linux-any (they only have i386/linux-i386 as overlap)
        >>> arch_table.matches_architecture("any-i386", "linux-any")
        False
        >>> # Compatibility with dpkg - if alias is `any` then it always returns True
        >>> # even if the input otherwise would not make sense.
        >>> arch_table.matches_architecture("any-unknown", "any")
        True
        >>> # Another side effect of the dpkg compatibility
        >>> arch_table.matches_architecture("all", "any")
        True

        Compatibility note: The method emulates Dpkg::Arch::debarch_is function and therefore
        returns True if both parameters are the same even though they are wildcards or not known
        to be architectures. Additionally, if `alias` is `any`, then this method always returns
        True as `any` is the "match-everything-wildcard".

        :param architecture: A string representing a dpkg architecture.
        :param alias: A string representing a dpkg architecture or wildcard
               to match with.
        :returns: True if the `architecture` parameter is (logically) the same as the `alias`
                  parameter OR, if `alias` is a wildcard, the `architecture` parameter is a
                  subset of the wildcard.
                  The method returns False if `architecture` is not a known dpkg architecture,
                  or it is a wildcard.
        """
    if alias in ('any', architecture):
        return True
    try:
        dpkg_arch = self._dpkg_arch_to_tuple(architecture)
        dpkg_wildcard = self._dpkg_wildcard_to_tuple(alias)
    except KeyError:
        return False
    return dpkg_arch in dpkg_wildcard