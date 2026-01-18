from .cartan_type import Standard_Cartan
from sympy.core.backend import eye, Rational
def dynkin_diagram(self):
    n = self.n
    diag = ' ' * 8 + str(2) + '\n'
    diag += ' ' * 8 + '0\n'
    diag += ' ' * 8 + '|\n'
    diag += ' ' * 8 + '|\n'
    diag += '---'.join(('0' for i in range(1, n))) + '\n'
    diag += '1   ' + '   '.join((str(i) for i in range(3, n + 1)))
    return diag