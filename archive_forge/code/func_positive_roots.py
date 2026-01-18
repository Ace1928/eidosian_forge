from .cartan_type import Standard_Cartan
from sympy.core.backend import eye, Rational
def positive_roots(self):
    """
        This method generates all the positive roots of
        A_n.  This is half of all of the roots of E_n;
        by multiplying all the positive roots by -1 we
        get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
        """
    n = self.n
    if n == 6:
        posroots = {}
        k = 0
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                k += 1
                root = self.basic_root(i, j)
                posroots[k] = root
                k += 1
                root = self.basic_root(i, j)
                root[i] = 1
                posroots[k] = root
        root = [Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]
        for a in range(0, 2):
            for b in range(0, 2):
                for c in range(0, 2):
                    for d in range(0, 2):
                        for e in range(0, 2):
                            if (a + b + c + d + e) % 2 == 0:
                                k += 1
                                if a == 1:
                                    root[0] = Rational(-1, 2)
                                if b == 1:
                                    root[1] = Rational(-1, 2)
                                if c == 1:
                                    root[2] = Rational(-1, 2)
                                if d == 1:
                                    root[3] = Rational(-1, 2)
                                if e == 1:
                                    root[4] = Rational(-1, 2)
                                posroots[k] = root
        return posroots
    if n == 7:
        posroots = {}
        k = 0
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                k += 1
                root = self.basic_root(i, j)
                posroots[k] = root
                k += 1
                root = self.basic_root(i, j)
                root[i] = 1
                posroots[k] = root
        k += 1
        posroots[k] = [0, 0, 0, 0, 0, 1, 1, 0]
        root = [Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]
        for a in range(0, 2):
            for b in range(0, 2):
                for c in range(0, 2):
                    for d in range(0, 2):
                        for e in range(0, 2):
                            for f in range(0, 2):
                                if (a + b + c + d + e + f) % 2 == 0:
                                    k += 1
                                    if a == 1:
                                        root[0] = Rational(-1, 2)
                                    if b == 1:
                                        root[1] = Rational(-1, 2)
                                    if c == 1:
                                        root[2] = Rational(-1, 2)
                                    if d == 1:
                                        root[3] = Rational(-1, 2)
                                    if e == 1:
                                        root[4] = Rational(-1, 2)
                                    if f == 1:
                                        root[5] = Rational(1, 2)
                                    posroots[k] = root
        return posroots
    if n == 8:
        posroots = {}
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                k += 1
                root = self.basic_root(i, j)
                posroots[k] = root
                k += 1
                root = self.basic_root(i, j)
                root[i] = 1
                posroots[k] = root
        root = [Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]
        for a in range(0, 2):
            for b in range(0, 2):
                for c in range(0, 2):
                    for d in range(0, 2):
                        for e in range(0, 2):
                            for f in range(0, 2):
                                for g in range(0, 2):
                                    if (a + b + c + d + e + f + g) % 2 == 0:
                                        k += 1
                                        if a == 1:
                                            root[0] = Rational(-1, 2)
                                        if b == 1:
                                            root[1] = Rational(-1, 2)
                                        if c == 1:
                                            root[2] = Rational(-1, 2)
                                        if d == 1:
                                            root[3] = Rational(-1, 2)
                                        if e == 1:
                                            root[4] = Rational(-1, 2)
                                        if f == 1:
                                            root[5] = Rational(1, 2)
                                        if g == 1:
                                            root[6] = Rational(1, 2)
                                        posroots[k] = root
        return posroots