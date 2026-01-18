from __future__ import absolute_import, division, print_function
import sys
def quick_is_not_prime(n):
    """Does some quick checks to see if we can poke a hole into the primality of n.

    A result of `False` does **not** mean that the number is prime; it just means
    that we could not detect quickly whether it is not prime.
    """
    if n <= 2:
        return True
    if simple_gcd(n, 7799922041683461553249199106329813876687996789903550945093032474868511536164700810) > 1:
        return True
    return False