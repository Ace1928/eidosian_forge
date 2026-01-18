from __future__ import absolute_import, division, print_function, unicode_literals
import codecs
from collections import defaultdict
from math import ceil, log as logf
import logging; log = logging.getLogger(__name__)
import pkg_resources
import os
from passlib import exc
from passlib.utils.compat import PY2, irange, itervalues, int_types
from passlib.utils import rng, getrandstr, to_unicode
from passlib.utils.decor import memoized_property
def genphrase(entropy=None, length=None, returns=None, **kwds):
    """Generate one or more random password / passphrases.

    This function uses :mod:`random.SystemRandom` to generate
    one or more passwords; it can be configured to generate
    alphanumeric passwords, or full english phrases.
    The complexity of the password can be specified
    by size, or by the desired amount of entropy.

    Usage Example::

        >>> # generate random phrase with 48 bits of entropy
        >>> from passlib import pwd
        >>> pwd.genphrase()
        'gangly robbing salt shove'

        >>> # generate a random phrase with 52 bits of entropy
        >>> # using a particular wordset
        >>> pwd.genword(entropy=52, wordset="bip39")
        'wheat dilemma reward rescue diary'

    :param entropy:
        Strength of resulting password, measured in 'guessing entropy' bits.
        An appropriate **length** value will be calculated
        based on the requested entropy amount, and the size of the word set.

        This can be a positive integer, or one of the following preset
        strings: ``"weak"`` (24), ``"fair"`` (36),
        ``"strong"`` (48), and ``"secure"`` (56).

        If neither this or **length** is specified, **entropy** will default
        to ``"strong"`` (48).

    :param length:
        Length of resulting password, measured in words.
        If omitted, the size is auto-calculated based on the **entropy** parameter.

        If both **entropy** and **length** are specified,
        the stronger value will be used.

    :param returns:
        Controls what this function returns:

        * If ``None`` (the default), this function will generate a single password.
        * If an integer, this function will return a list containing that many passwords.
        * If the ``iter`` builtin, will return an iterator that yields passwords.

    :param words:

        Optionally specifies a list/set of words to use when randomly generating a passphrase.
        This option cannot be combined with **wordset**.

    :param wordset:

        The predefined word set to draw from (if not specified by **words**).
        There are currently four presets available:

        ``"eff_long"`` (the default)

            Wordset containing 7776 english words of ~7 letters.
            Constructed by the EFF, it offers ~12.9 bits of entropy per word.

            This wordset (and the other ``"eff_"`` wordsets)
            were `created by the EFF <https://www.eff.org/deeplinks/2016/07/new-wordlists-random-passphrases>`_
            to aid in generating passwords.  See their announcement page
            for more details about the design & properties of these wordsets.

        ``"eff_short"``

            Wordset containing 1296 english words of ~4.5 letters.
            Constructed by the EFF, it offers ~10.3 bits of entropy per word.

        ``"eff_prefixed"``

            Wordset containing 1296 english words of ~8 letters,
            selected so that they each have a unique 3-character prefix.
            Constructed by the EFF, it offers ~10.3 bits of entropy per word.

        ``"bip39"``

            Wordset of 2048 english words of ~5 letters,
            selected so that they each have a unique 4-character prefix.
            Published as part of Bitcoin's `BIP 39 <https://github.com/bitcoin/bips/blob/master/bip-0039/english.txt>`_,
            this wordset has exactly 11 bits of entropy per word.

            This list offers words that are typically shorter than ``"eff_long"``
            (at the cost of slightly less entropy); and much shorter than
            ``"eff_prefixed"`` (at the cost of a longer unique prefix).

    :param sep:
        Optional separator to use when joining words.
        Defaults to ``" "`` (a space), but can be an empty string, a hyphen, etc.

    :returns:
        :class:`!unicode` string containing randomly generated passphrase;
        or list of 1+ passphrases if :samp:`returns={int}` is specified.
    """
    gen = PhraseGenerator(entropy=entropy, length=length, **kwds)
    return gen(returns)