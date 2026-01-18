from rsa._compat import range
import rsa.common
import rsa.randnum
def get_primality_testing_rounds(number):
    """Returns minimum number of rounds for Miller-Rabing primality testing,
    based on number bitsize.

    According to NIST FIPS 186-4, Appendix C, Table C.3, minimum number of
    rounds of M-R testing, using an error probability of 2 ** (-100), for
    different p, q bitsizes are:
      * p, q bitsize: 512; rounds: 7
      * p, q bitsize: 1024; rounds: 4
      * p, q bitsize: 1536; rounds: 3
    See: http://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf
    """
    bitsize = rsa.common.bit_size(number)
    if bitsize >= 1536:
        return 3
    if bitsize >= 1024:
        return 4
    if bitsize >= 512:
        return 7
    return 10