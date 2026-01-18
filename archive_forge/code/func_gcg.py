import binascii
def gcg(seq):
    """Return the GCG checksum (int) for a sequence (string or Seq object).

    Given a nucleotide or amino-acid sequence (or any string),
    returns the GCG checksum (int). Checksum used by GCG program.
    seq type = str.

    Based on BioPerl GCG_checksum. Adapted by Sebastian Bassi
    with the help of John Lenton, Pablo Ziliani, and Gabriel Genellina.

    All sequences are converted to uppercase.

    >>> gcg("ACGTACGTACGT")
    5688
    >>> gcg("acgtACGTacgt")
    5688

    """
    index = checksum = 0
    for char in seq:
        index += 1
        checksum += index * ord(char.upper())
        if index == 57:
            index = 0
    return checksum % 10000