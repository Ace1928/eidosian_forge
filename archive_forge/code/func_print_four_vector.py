from ase.lattice.triclinic import TriclinicFactory
def print_four_vector(self, bracket, numbers):
    bra, ket = bracket
    x, y, z = numbers
    a = 2 * x - y
    b = -x + 2 * y
    c = -x - y
    d = 2 * z
    print('   %s%d, %d, %d%s  ~  %s%d, %d, %d, %d%s' % (bra, x, y, z, ket, bra, a, b, c, d, ket))