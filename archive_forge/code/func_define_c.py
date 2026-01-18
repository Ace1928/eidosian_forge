from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def define_c(self, alpha, x):
    """
        A variation of ``define`` routine, described on Pg. 165 [1], used in
        the coset table-based strategy of Todd-Coxeter algorithm. It differs
        from ``define`` routine in that for each definition it also adds the
        tuple `(\\alpha, x)` to the deduction stack.

        See Also
        ========

        define

        """
    A = self.A
    table = self.table
    len_table = len(table)
    if len_table >= self.coset_table_limit:
        raise ValueError('the coset enumeration has defined more than %s cosets. Try with a greater value max number of cosets ' % self.coset_table_limit)
    table.append([None] * len(A))
    beta = len_table
    self.p.append(beta)
    table[alpha][self.A_dict[x]] = beta
    table[beta][self.A_dict_inv[x]] = alpha
    self.deduction_stack.append((alpha, x))