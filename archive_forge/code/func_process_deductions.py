from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def process_deductions(self, R_c_x, R_c_x_inv):
    """
        Processes the deductions that have been pushed onto ``deduction_stack``,
        described on Pg. 166 [1] and is used in coset-table based enumeration.

        See Also
        ========

        deduction_stack

        """
    p = self.p
    table = self.table
    while len(self.deduction_stack) > 0:
        if len(self.deduction_stack) >= CosetTable.max_stack_size:
            self.look_ahead()
            del self.deduction_stack[:]
            continue
        else:
            alpha, x = self.deduction_stack.pop()
            if p[alpha] == alpha:
                for w in R_c_x:
                    self.scan_c(alpha, w)
                    if p[alpha] < alpha:
                        break
        beta = table[alpha][self.A_dict[x]]
        if beta is not None and p[beta] == beta:
            for w in R_c_x_inv:
                self.scan_c(beta, w)
                if p[beta] < beta:
                    break