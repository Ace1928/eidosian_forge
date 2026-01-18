import os
import collections.abc
def architecture_equals(self, arch1, arch2):
    """Determine whether two dpkg architecture are exactly the same [debarch_eq]

        Unlike Python's `==` operator, this method also accounts for things like `linux-amd64` is
        a valid spelling of the dpkg architecture `amd64` (i.e.,
        `architecture_equals("linux-amd64", "amd64")` is True).

        This method is the closest match to dpkg's Dpkg::Arch::debarch_eq function.

        >>> arch_table = DpkgArchTable.load_arch_table()
        >>> arch_table.architecture_equals("linux-amd64", "amd64")
        True
        >>> arch_table.architecture_equals("amd64", "linux-i386")
        False
        >>> arch_table.architecture_equals("i386", "linux-amd64")
        False
        >>> arch_table.architecture_equals("amd64", "amd64")
        True
        >>> arch_table.architecture_equals("i386", "amd64")
        False
        >>> # Compatibility with dpkg: if the parameters are equal, then it always return True
        >>> arch_table.architecture_equals("unknown", "unknown")
        True

        Compatibility note: The method emulates Dpkg::Arch::debarch_eq function and therefore
        returns True if both parameters are the same even though they are wildcards or not known
        to be architectures.

        :param arch1: A string representing a dpkg architecture.
        :param arch2: A string representing a dpkg architecture.
        :returns: True if the dpkg architecture parameters are (logically) the exact same.
        """
    if arch1 == arch2:
        return True
    try:
        dpkg_arch1 = self._dpkg_arch_to_tuple(arch1)
        dpkg_arch2 = self._dpkg_arch_to_tuple(arch2)
    except KeyError:
        return False
    return dpkg_arch1 == dpkg_arch2